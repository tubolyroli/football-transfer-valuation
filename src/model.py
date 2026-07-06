"""Train and compare regression models on the merged football dataset.

Key choices:
- Target is log1p(value_eur): market value is heavily right-skewed (raw skew ≈ 2.6, log ≈ 1.1).
- Position is collapsed to primary role (first listed) and one-hot encoded.
- League is one-hot encoded so the model can learn league-level premiums (the "EPL Tax").
- Hyperparameters are selected with GridSearchCV on the training split only —
  the 20% test set is never seen during tuning.
- League premium and undervalued rankings use out-of-fold predictions
  (cross_val_predict), so no player is scored by a model that saw them in training.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_predict, train_test_split
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

PARAM_GRIDS = {
    "Ridge": {
        "model__alpha": np.logspace(-2, 3, 11),
    },
    "RandomForest": {
        "model__max_depth": [None, 5, 10],
        "model__min_samples_leaf": [1, 5, 10],
        "model__max_features": ["sqrt", 0.5, 1.0],
    },
    "GradientBoosting": {
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__n_estimators": [100, 300],
        "model__max_depth": [2, 3],
    },
}


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
        "Ridge": Pipeline([("pre", pre()), ("model", Ridge())]),
        "RandomForest": Pipeline(
            [("pre", pre()), ("model", RandomForestRegressor(n_estimators=300, random_state=42))]
        ),
        "GradientBoosting": Pipeline(
            [("pre", pre()), ("model", GradientBoostingRegressor(random_state=42))]
        ),
    }


def tune_and_evaluate(name, pipe, grid, X_train, X_test, y_train_log, y_test_raw) -> dict:
    """Grid-search hyperparameters on the training split, then score on the test set.

    Selection metric is 5-fold CV R² in log space (same space the model optimizes);
    reported test metrics are computed in € after inverting the log transform.
    """
    search = GridSearchCV(pipe, grid, cv=5, scoring="r2", n_jobs=-1)
    search.fit(X_train, y_train_log)

    best = search.best_estimator_
    test_pred = np.expm1(best.predict(X_test))
    r2 = r2_score(y_test_raw, test_pred)
    mae = mean_absolute_error(y_test_raw, test_pred)

    cv_mean = search.best_score_
    cv_std = search.cv_results_["std_test_score"][search.best_index_]

    params = {k.removeprefix("model__"): v for k, v in search.best_params_.items()}
    print(f"\n--- {name} ---")
    print(f"Best params:               {params}")
    print(f"5-fold CV R² (log, train): {cv_mean:.3f} ± {cv_std:.3f}")
    print(f"Held-out R² (on €):        {r2:.3f}")
    print(f"Held-out MAE:              €{mae:,.0f}")
    return {
        "name": name,
        "r2": r2,
        "mae": mae,
        "cv_r2": cv_mean,
        "best_params": params,
        "estimator": best,
    }


def add_oof_predictions(df: pd.DataFrame, pipe: Pipeline, X, y_log) -> pd.DataFrame:
    """Attach out-of-fold predictions: each player is scored by a model fold that never trained on them."""
    oof_log = cross_val_predict(clone(pipe), X, y_log, cv=5)
    df = df.copy()
    df["predicted_value"] = np.expm1(oof_log)
    df["difference"] = df["predicted_value"] - df[TARGET]
    df["price_premium_pct"] = (df[TARGET] / df["predicted_value"] - 1) * 100
    return df


def league_tax_report(df: pd.DataFrame) -> None:
    """Quantify how much extra each league adds vs. the cross-league baseline (out-of-fold)."""
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
    print("\n--- League Price Premium, out-of-fold (positive = priced above model expectation) ---")
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
        results.append(
            tune_and_evaluate(name, pipe, PARAM_GRIDS[name], X_train, X_test, y_train_log, y_test_raw)
        )

    leaderboard = pd.DataFrame(results).drop(columns="estimator").sort_values("r2", ascending=False)
    print("\n=== Leaderboard (sorted by held-out R²) ===")
    print(leaderboard.to_string(index=False))

    best_name = leaderboard.iloc[0]["name"]
    best_pipe = next(r["estimator"] for r in results if r["name"] == best_name)
    print(f"\n[BEST] {best_name}")

    scored = add_oof_predictions(df, best_pipe, X, y_log)
    league_tax_report(scored)

    undervalued = scored.sort_values("difference", ascending=False).head(5)
    print(f"\n--- Top 5 Undervalued (Out-of-Fold, {best_name}) ---")
    print(undervalued[["player_name", "league", "value_eur", "predicted_value", "difference"]])

    return best_pipe, scored


if __name__ == "__main__":
    train_model()
