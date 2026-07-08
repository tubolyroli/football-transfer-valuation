import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from formatting import fmt_eur

NUMERIC_FEATURES = [
    "age_clean", "goals", "assists", "xg", "Expected_xAG",
    "Progression_PrgC", "Playing Time_Min",
]
CATEGORICAL_FEATURES = ["position_primary", "league"]
TARGET = "value_eur"
ALPHA_GRID = np.logspace(-2, 3, 11)

REPO_URL = "https://github.com/tubolyroli/football-transfer-valuation"

# Display names for FBref's prefixed league strings. The model keeps the raw
# values — only what the user sees is relabeled.
LEAGUE_LABELS = {
    "de Bundesliga": "Bundesliga",
    "eng Premier League": "Premier League",
    "es La Liga": "La Liga",
    "fr Ligue 1": "Ligue 1",
    "it Serie A": "Serie A",
}
FEATURE_LABELS = {
    "age_clean": "Age",
    "goals": "Goals",
    "assists": "Assists",
    "xg": "xG",
    "Expected_xAG": "xAG",
    "Progression_PrgC": "Progressive carries",
    "Playing Time_Min": "Minutes played",
}

# Colors follow the entity, not the filter order: each league keeps its slot
# even when others are filtered out. Palette validated for CVD (all pairs).
LEAGUE_COLORS = {
    "Bundesliga": "#2a78d6",
    "La Liga": "#1baf7a",
    "Ligue 1": "#eda100",
    "Premier League": "#008300",
    "Serie A": "#4a3aa7",
}
LEAGUE_ORDER = list(LEAGUE_COLORS)
BLUE, RED, MUTED = "#2a78d6", "#e34948", "#898781"
# Diverging: red = overpriced, gray = fairly priced, blue = undervalued.
DIVERGING_SCALE = [(0.0, RED), (0.5, "#f0efec"), (1.0, BLUE)]

st.set_page_config(page_title="Football Transfer Valuation", page_icon="⚽", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/final_dataset.csv")
    df["age_clean"] = df["age"].astype(str).str.split("-").str[0].astype(int)
    df["position_primary"] = df["position"].astype(str).str.split(",").str[0]
    df["log_value"] = np.log1p(df[TARGET])
    df["league_display"] = df["league"].map(LEAGUE_LABELS).fillna(df["league"])
    return df


def build_pipeline():
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("pre", pre), ("model", Ridge())])


def prettify_feature(name: str) -> str:
    """Human-readable label for a raw or one-hot-encoded feature name."""
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    if name.startswith("position_primary_"):
        return "Position: " + name.removeprefix("position_primary_")
    if name.startswith("league_"):
        raw = name.removeprefix("league_")
        return "League: " + LEAGUE_LABELS.get(raw, raw)
    return name


@st.cache_data
def train_and_predict(df: pd.DataFrame):
    """Tune alpha on the training split, score on the test split, and attach
    out-of-fold predictions so every displayed ranking is out-of-sample."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_log = df["log_value"]
    y_raw = df[TARGET]

    X_train, X_test, y_train_log, _, _, y_test_raw = train_test_split(
        X, y_log, y_raw, test_size=0.2, random_state=42
    )

    search = GridSearchCV(build_pipeline(), {"model__alpha": ALPHA_GRID}, cv=5, scoring="r2")
    search.fit(X_train, y_train_log)
    pipe = search.best_estimator_
    best_alpha = search.best_params_["model__alpha"]

    test_pred = np.expm1(pipe.predict(X_test))
    r2 = r2_score(y_test_raw, test_pred)
    mae = mean_absolute_error(y_test_raw, test_pred)

    # Out-of-fold predictions: each player is scored by a fold that never trained on them.
    oof_log = cross_val_predict(clone(pipe), X, y_log, cv=5)
    df = df.copy()
    df["predicted_value"] = np.expm1(oof_log)
    df["difference"] = df["predicted_value"] - df[TARGET]
    df["residual"] = df[TARGET] - df["predicted_value"]

    # Feature importance: standardized coefficients from the linear model.
    ohe = pipe.named_steps["pre"].named_transformers_["cat"]
    feature_names = list(NUMERIC_FEATURES) + list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))
    coefs = pipe.named_steps["model"].coef_
    importance = pd.DataFrame({"feature": feature_names, "coefficient": coefs})
    importance["abs"] = importance["coefficient"].abs()
    importance = importance.sort_values("abs", ascending=True)

    return df, r2, mae, importance, best_alpha


def money_axes(fig):
    fig.update_xaxes(tickprefix="€", tickformat="~s")
    fig.update_yaxes(tickprefix="€", tickformat="~s")
    return fig


def money_column(label: str) -> "st.column_config.NumberColumn":
    """Column config for a value already converted to € millions."""
    return st.column_config.NumberColumn(label, format="€%.1fM")


df = load_data()

# Sidebar filters
st.sidebar.header("Filter Players")
selected_leagues = st.sidebar.multiselect(
    "Leagues",
    options=sorted(df["league"].dropna().unique()),
    default=sorted(df["league"].dropna().unique()),
    format_func=lambda l: LEAGUE_LABELS.get(l, l),
)
min_minutes = st.sidebar.slider("Minimum Minutes Played", 0, 3000, 500)

df_view = df[
    (df["league"].isin(selected_leagues)) & (df["Playing Time_Min"] >= min_minutes)
].copy()

if len(df_view) < 20:
    st.warning("Not enough players match the filters to train a model. Loosen the filters.")
    st.stop()

with st.spinner(f"Tuning and training Ridge on {len(df_view)} players…"):
    scored, r2, mae, importance, best_alpha = train_and_predict(df_view)

# Header + headline metrics
st.title("⚽ Football Transfer Valuation Model")
st.markdown(
    "Ridge regression on FBref performance stats + position + league predicts "
    "**log(market value)**. Positive *difference* = the model thinks the player is **undervalued**."
)
st.caption(
    f"Data: FBref Big-5 league stats × Transfermarkt top-500 valuations · "
    f"[Source on GitHub]({REPO_URL})"
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Players", len(scored))
col2.metric("Held-out R²", f"{r2:.2f}")
col3.metric("Held-out MAE", fmt_eur(mae))
col4.metric("Median Market Value", fmt_eur(scored[TARGET].median()))

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Predicted vs Actual", "📉 Residuals", "🏆 League Premium", "🔬 Feature Importance"]
)

with tab1:
    st.subheader("Actual vs. Predicted Market Value")
    fig = px.scatter(
        scored, x="value_eur", y="predicted_value",
        custom_data=["player_name", "team", "league_display"],
        color="difference", color_continuous_scale=DIVERGING_SCALE,
        color_continuous_midpoint=0,
        labels={"value_eur": "Actual market value", "predicted_value": "Model prediction"},
    )
    fig.update_traces(
        marker=dict(size=8, line=dict(width=1, color="rgba(11,11,11,0.10)")),
        hovertemplate="<b>%{customdata[0]}</b> — %{customdata[1]} (%{customdata[2]})"
                      "<br>Actual €%{x:.3s} · Predicted €%{y:.3s}<extra></extra>",
    )
    fig.update_coloraxes(colorbar=dict(title="Δ (pred − actual)", tickprefix="€", tickformat="~s"))
    max_val = float(scored[["value_eur", "predicted_value"]].max().max())
    fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                  line=dict(color=MUTED, dash="dash"))
    st.plotly_chart(money_axes(fig), width="stretch")

    st.subheader("💎 Top 10 Undervalued (Out-of-Fold Predictions)")
    st.caption(
        "Every player's prediction comes from a cross-validation fold that never saw them "
        "in training, so the whole table is out-of-sample — not just a 20% test slice."
    )
    undervalued = scored.sort_values("difference", ascending=False).head(10)
    table = undervalued[["player_name", "age_clean", "team", "league_display",
                         "value_eur", "predicted_value", "difference"]].copy()
    for col in ["value_eur", "predicted_value", "difference"]:
        table[col] /= 1e6
    st.dataframe(
        table,
        hide_index=True,
        column_config={
            "player_name": "Player",
            "age_clean": st.column_config.NumberColumn("Age", format="%d"),
            "team": "Team",
            "league_display": "League",
            "value_eur": money_column("Actual"),
            "predicted_value": money_column("Predicted"),
            "difference": st.column_config.ProgressColumn(
                "Undervalued by", format="€%.1fM",
                min_value=0.0, max_value=float(table["difference"].max()),
            ),
        },
    )

with tab2:
    st.subheader("Residual Plot")
    st.caption(
        "Residual = Actual − Predicted. Random scatter around zero means the model is well-specified. "
        "Patterns (e.g., funnel shape) reveal heteroscedasticity or missing features."
    )
    fig = px.scatter(
        scored, x="predicted_value", y="residual",
        custom_data=["player_name", "team"],
        color="league_display", color_discrete_map=LEAGUE_COLORS,
        category_orders={"league_display": LEAGUE_ORDER},
        labels={"predicted_value": "Model prediction", "residual": "Residual",
                "league_display": "League"},
    )
    fig.update_traces(
        marker=dict(size=8),
        hovertemplate="<b>%{customdata[0]}</b> — %{customdata[1]}"
                      "<br>Predicted €%{x:.3s} · Residual €%{y:.3s}<extra></extra>",
    )
    fig.add_hline(y=0, line_dash="dash", line_color=MUTED)
    st.plotly_chart(money_axes(fig), width="stretch")

    with st.expander("View the underlying data"):
        resid_table = scored[["player_name", "team", "league_display",
                              "value_eur", "predicted_value", "residual"]].copy()
        resid_table = resid_table.sort_values("residual")
        for col in ["value_eur", "predicted_value", "residual"]:
            resid_table[col] /= 1e6
        st.dataframe(
            resid_table,
            hide_index=True,
            column_config={
                "player_name": "Player",
                "team": "Team",
                "league_display": "League",
                "value_eur": money_column("Actual"),
                "predicted_value": money_column("Predicted"),
                "residual": money_column("Residual"),
            },
        )

with tab3:
    st.subheader("League Price Premium")
    st.caption(
        "For each league: median % difference between actual market value and the model's prediction. "
        "Positive = league players are priced *above* what their stats predict (the 'tax')."
    )
    league_summary = (
        scored.assign(premium_pct=(scored[TARGET] / scored["predicted_value"] - 1) * 100)
        .groupby("league_display")
        .agg(
            n=("player_name", "size"),
            median_value=(TARGET, "median"),
            median_predicted=("predicted_value", "median"),
            median_premium_pct=("premium_pct", "median"),
        )
        .sort_values("median_premium_pct", ascending=False)
        .reset_index()
    )
    league_summary["direction"] = np.where(
        league_summary["median_premium_pct"] >= 0, "Priced above stats", "Priced below stats"
    )
    fig = px.bar(
        league_summary, x="league_display", y="median_premium_pct",
        color="direction",
        color_discrete_map={"Priced above stats": RED, "Priced below stats": BLUE},
        labels={"median_premium_pct": "Median premium", "league_display": "", "direction": ""},
    )
    fig.update_traces(hovertemplate="<b>%{x}</b><br>Median premium %{y:.1f}%<extra></extra>")
    fig.update_yaxes(ticksuffix="%")
    fig.update_layout(barcornerradius=4)
    fig.add_hline(y=0, line_dash="dash", line_color=MUTED)
    st.plotly_chart(fig, width="stretch")

    summary_table = league_summary.drop(columns="direction").copy()
    for col in ["median_value", "median_predicted"]:
        summary_table[col] /= 1e6
    st.dataframe(
        summary_table,
        hide_index=True,
        column_config={
            "league_display": "League",
            "n": st.column_config.NumberColumn("Players", format="%d"),
            "median_value": money_column("Median actual"),
            "median_predicted": money_column("Median predicted"),
            "median_premium_pct": st.column_config.NumberColumn(
                "Median premium", format="%.1f%%"
            ),
        },
    )

with tab4:
    st.subheader("Standardized Ridge Coefficients")
    st.caption(
        "Numeric features are standardized before fitting, so coefficient magnitudes are directly comparable. "
        "Categorical (league/position) coefficients are relative to the implicit baseline category. "
        "Blue raises the predicted value, red lowers it."
    )
    fig = go.Figure(go.Bar(
        x=importance["coefficient"],
        y=importance["feature"].map(prettify_feature),
        orientation="h",
        marker=dict(color=np.where(importance["coefficient"] >= 0, BLUE, RED)),
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(height=600, xaxis_title="Coefficient (log-€ units)", barcornerradius=4)
    st.plotly_chart(fig, width="stretch")

st.markdown("### 📝 Methodology")
st.markdown(
    f"""
- **Target:** `log1p(value_eur)` — raw market value is right-skewed (raw skew ≈ 2.6, log ≈ 1.1).
- **Numeric features:** {', '.join(NUMERIC_FEATURES)} (standardized).
- **Categorical features:** {', '.join(CATEGORICAL_FEATURES)} (one-hot encoded).
- **Tuning:** Ridge `alpha` selected by 5-fold `GridSearchCV` on the training split only
  (log-spaced grid 10⁻²–10³; chosen for the current filter: `alpha = {best_alpha:g}`).
- **Validation:** 80/20 train/test split, fixed `random_state=42`. R² and MAE reported on the
  held-out 20% only. Rankings and league premiums use out-of-fold (`cross_val_predict`) predictions.
- **Limitations:** No contract length, no commercial/marketing value, no transfermarkt-specific market sentiment. Sample is the top-500 most valuable players only — selection bias toward expensive players.
"""
)
