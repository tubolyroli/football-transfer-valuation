"""Train and compare regression models on the merged football dataset.

Key choices:
- Target is log1p(value_eur): market value is heavily right-skewed (raw skew ≈ 2.6, log ≈ 1.1).
- Position is collapsed to primary role (first listed) and one-hot encoded.
- League is one-hot encoded so the model can learn league-level premiums (the "EPL Tax").
- Three models are compared on the same train/test split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "age_clean",
    "goals",
    "assists",
    "xg",
    "Expected_xAG",
    "Progression_PrgC",
    "Playing Time_Min",
]
CATEGORICAL_FEATURES = ["position_primary", "league"]
TARGET = "value_eur"


def load_dataset(path: str = "data/processed/final_dataset.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    df["age_clean"] = df["age"].astype(str).str.split("-").str[0].astype(int)
    df["position_primary"] = df["position"].astype(str).str.split(",").str[0]
    df["log_value"] = np.log1p(df[TARGET])
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def build_models() -> dict[str, Pipeline]:
    pre = build_preprocessor
    return {
        "Ridge": Pipeline([("pre", pre()), ("model", Ridge(alpha=1.0))]),
        "RandomForest": Pipeline(
            [("pre", pre()), ("model", RandomForestRegressor(n_estimators=300, random_state=42))]
        ),
        "GradientBoosting": Pipeline(
            [("pre", pre()), ("model", GradientBoostingRegressor(random_state=42))]
        ),
    }


def evaluate(name: str, pipe: Pipeline, X_train, X_test, y_train_log, y_test_raw, X_all, y_all_log):
    pipe.fit(X_train, y_train_log)

    # Predictions are in log-space; invert back to € for honest MAE.
    test_log_pred = pipe.predict(X_test)
    test_pred = np.expm1(test_log_pred)
    r2 = r2_score(y_test_raw, test_pred)
    mae = mean_absolute_error(y_test_raw, test_pred)

    cv_r2 = cross_val_score(pipe, X_all, y_all_log, cv=5, scoring="r2")

    print(f"\n--- {name} ---")
    print(f"Held-out R² (on €):  {r2:.3f}")
    print(f"Held-out MAE:        €{mae:,.0f}")
    print(f"5-fold CV R² (log):  {cv_r2.mean():.3f} ± {cv_r2.std():.3f}")
    return {"name": name, "r2": r2, "mae": mae, "cv_r2": cv_r2.mean()}


def league_tax_report(df: pd.DataFrame, pipe: Pipeline, X) -> None:
    """Quantify how much extra each league adds vs. the cross-league baseline."""
    log_pred = pipe.predict(X)
    df = df.copy()
    df["predicted_value"] = np.expm1(log_pred)
    df["residual_eur"] = df[TARGET] - df["predicted_value"]
    df["price_premium_pct"] = (df[TARGET] / df["predicted_value"] - 1) * 100

    by_league = (
        df.groupby("league")
        .agg(
            n=("player_name", "size"),
            mean_value=(TARGET, "mean"),
            mean_predicted=("predicted_value", "mean"),
            median_premium_pct=("price_premium_pct", "median"),
        )
        .sort_values("median_premium_pct", ascending=False)
    )
    print("\n--- League Price Premium (positive = priced above model expectation) ---")
    print(by_league.round(2))


def train_model():
    df = load_dataset()

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_cols]
    y_log = df["log_value"]
    y_raw = df[TARGET]

    X_train, X_test, y_train_log, y_test_log, y_train_raw, y_test_raw = train_test_split(
        X, y_log, y_raw, test_size=0.2, random_state=42
    )

    models = build_models()
    results = []
    for name, pipe in models.items():
        results.append(evaluate(name, pipe, X_train, X_test, y_train_log, y_test_raw, X, y_log))

    leaderboard = pd.DataFrame(results).sort_values("r2", ascending=False)
    print("\n=== Leaderboard (sorted by held-out R²) ===")
    print(leaderboard.to_string(index=False))

    best_name = leaderboard.iloc[0]["name"]
    best_pipe = models[best_name]
    print(f"\n[BEST] {best_name}")

    league_tax_report(df, best_pipe, X)

    # Top out-of-sample undervalued picks from the best model.
    test_idx = X_test.index
    test_df = df.loc[test_idx].copy()
    test_df["predicted_value"] = np.expm1(best_pipe.predict(X_test))
    test_df["difference"] = test_df["predicted_value"] - test_df[TARGET]
    undervalued = test_df.sort_values("difference", ascending=False).head(5)
    print(f"\n--- Top 5 Undervalued (Out-of-Sample, {best_name}) ---")
    print(undervalued[["player_name", "league", "value_eur", "predicted_value", "difference"]])

    return best_pipe, df


if __name__ == "__main__":
    train_model()
