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

NUMERIC_FEATURES = [
    "age_clean", "goals", "assists", "xg", "Expected_xAG",
    "Progression_PrgC", "Playing Time_Min",
]
CATEGORICAL_FEATURES = ["position_primary", "league"]
TARGET = "value_eur"
ALPHA_GRID = np.logspace(-2, 3, 11)

st.set_page_config(page_title="Football Transfer Valuation", page_icon="⚽", layout="wide")


@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/final_dataset.csv")
    df["age_clean"] = df["age"].astype(str).str.split("-").str[0].astype(int)
    df["position_primary"] = df["position"].astype(str).str.split(",").str[0]
    df["log_value"] = np.log1p(df[TARGET])
    return df


def build_pipeline():
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline([("pre", pre), ("model", Ridge())])


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


df = load_data()

# Sidebar filters
st.sidebar.header("Filter Players")
selected_leagues = st.sidebar.multiselect(
    "Leagues",
    options=sorted(df["league"].dropna().unique()),
    default=sorted(df["league"].dropna().unique()),
)
min_minutes = st.sidebar.slider("Minimum Minutes Played", 0, 3000, 500)

df_view = df[
    (df["league"].isin(selected_leagues)) & (df["Playing Time_Min"] >= min_minutes)
].copy()

if len(df_view) < 20:
    st.warning("Not enough players match the filters to train a model. Loosen the filters.")
    st.stop()

scored, r2, mae, importance, best_alpha = train_and_predict(df_view)

# Header + headline metrics
st.title("⚽ Football Transfer Valuation Model")
st.markdown(
    "Ridge regression on FBref performance stats + position + league predicts "
    "**log(market value)**. Positive *difference* = the model thinks the player is **undervalued**."
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Players", len(scored))
col2.metric("Held-out R²", f"{r2:.2%}")
col3.metric("Held-out MAE", f"€{mae:,.0f}")
col4.metric("Top League by N", scored["league"].value_counts().idxmax())

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Predicted vs Actual", "📉 Residuals", "🏆 League Premium", "🔬 Feature Importance"]
)

with tab1:
    st.subheader("Actual vs. Predicted Market Value")
    fig = px.scatter(
        scored, x="value_eur", y="predicted_value",
        hover_data=["player_name", "age", "team", "league"],
        color="difference", color_continuous_scale="RdBu",
        labels={"value_eur": "Actual (€)", "predicted_value": "Predicted (€)"},
    )
    max_val = float(scored[["value_eur", "predicted_value"]].max().max())
    fig.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                  line=dict(color="green", dash="dash"))
    st.plotly_chart(fig, width="stretch")

    st.subheader("💎 Top 10 Undervalued (Out-of-Fold Predictions)")
    st.caption(
        "Every player's prediction comes from a cross-validation fold that never saw them "
        "in training, so the whole table is out-of-sample — not just a 20% test slice."
    )
    undervalued = scored.sort_values("difference", ascending=False).head(10)
    st.dataframe(
        undervalued[["player_name", "age", "team", "league",
                     "value_eur", "predicted_value", "difference"]]
        .style.format({"value_eur": "€{:,.0f}",
                       "predicted_value": "€{:,.0f}",
                       "difference": "€{:,.0f}"})
    )

with tab2:
    st.subheader("Residual Plot")
    st.caption(
        "Residual = Actual − Predicted. Random scatter around zero means the model is well-specified. "
        "Patterns (e.g., funnel shape) reveal heteroscedasticity or missing features."
    )
    fig = px.scatter(
        scored, x="predicted_value", y="residual",
        hover_data=["player_name", "league"],
        color="league",
        labels={"predicted_value": "Predicted (€)", "residual": "Residual (€)"},
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    st.plotly_chart(fig, width="stretch")

with tab3:
    st.subheader("League Price Premium")
    st.caption(
        "For each league: median % difference between actual market value and the model's prediction. "
        "Positive = league players are priced *above* what their stats predict (the 'tax')."
    )
    league_summary = (
        scored.assign(premium_pct=(scored[TARGET] / scored["predicted_value"] - 1) * 100)
        .groupby("league")
        .agg(
            n=("player_name", "size"),
            median_value=(TARGET, "median"),
            median_predicted=("predicted_value", "median"),
            median_premium_pct=("premium_pct", "median"),
        )
        .sort_values("median_premium_pct", ascending=False)
        .reset_index()
    )
    fig = px.bar(
        league_summary, x="league", y="median_premium_pct",
        color="median_premium_pct", color_continuous_scale="RdBu_r",
        labels={"median_premium_pct": "Median premium (%)"},
        title="Are league players priced above or below their stats?",
    )
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(
        league_summary.style.format({
            "median_value": "€{:,.0f}",
            "median_predicted": "€{:,.0f}",
            "median_premium_pct": "{:.1f}%",
        })
    )

with tab4:
    st.subheader("Standardized Ridge Coefficients")
    st.caption(
        "Numeric features are standardized before fitting, so coefficient magnitudes are directly comparable. "
        "Categorical (league/position) coefficients are relative to the implicit baseline category."
    )
    fig = go.Figure(go.Bar(
        x=importance["coefficient"], y=importance["feature"],
        orientation="h",
        marker=dict(color=importance["coefficient"], colorscale="RdBu", cmid=0),
    ))
    fig.update_layout(height=600, xaxis_title="Coefficient (log-€ units)")
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
