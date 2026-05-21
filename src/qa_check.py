"""Data-quality validation for the merged dataset.

Runs Great Expectations checks against data/processed/final_dataset.csv and
exits non-zero if any expectation fails, so it can gate a CI pipeline.
"""

import sys
import pandas as pd
import great_expectations as gx

DATA_PATH = "data/processed/final_dataset.csv"

EXPECTED_LEAGUES = {
    "eng Premier League",
    "es La Liga",
    "de Bundesliga",
    "it Serie A",
    "fr Ligue 1",
}


def load_dataset():
    df = pd.read_csv(DATA_PATH)
    # FBref encodes age as "23-054"; keep just the year part for validation.
    df["age_years"] = df["age"].astype(str).str.split("-").str[0].astype(int)
    return df


def build_validator(df):
    context = gx.get_context()
    data_source = context.data_sources.add_pandas(name="football_valuation")
    asset = data_source.add_dataframe_asset(name="final_dataset")
    batch_definition = asset.add_batch_definition_whole_dataframe("all")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})
    return batch


def run_checks(batch):
    expectations = [
        gx.expectations.ExpectColumnValuesToNotBeNull(column="player_name"),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="value_eur"),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="value_eur", min_value=0, max_value=300_000_000
        ),
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="age_years", min_value=15, max_value=45
        ),
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="league", value_set=list(EXPECTED_LEAGUES)
        ),
        gx.expectations.ExpectColumnValuesToBeUnique(column="name_key"),
    ]

    failed = []
    for exp in expectations:
        result = batch.validate(exp)
        status = "PASS" if result.success else "FAIL"
        print(f"[{status}] {exp.__class__.__name__} on {exp.column}")
        if not result.success:
            unexpected = result.result.get("unexpected_count", "?")
            print(f"        Unexpected count: {unexpected}")
            failed.append(exp.__class__.__name__)
    return failed


def main():
    print(f"[INFO] Validating {DATA_PATH}...")
    df = load_dataset()
    print(f"[INFO] Loaded {len(df)} rows.")

    batch = build_validator(df)
    failed = run_checks(batch)

    if failed:
        print(f"\n[CRITICAL] {len(failed)} expectation(s) failed: {failed}")
        sys.exit(1)
    print("\n[SUCCESS] All data quality checks passed.")


if __name__ == "__main__":
    main()
