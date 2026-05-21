import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_cleaning import clean_money, normalize_name


class TestCleanMoney:
    def test_millions(self):
        assert clean_money("€200.00m") == 200_000_000

    def test_thousands(self):
        assert clean_money("€500k") == 500_000

    def test_no_symbol(self):
        assert clean_money("75m") == 75_000_000

    def test_uppercase(self):
        assert clean_money("€1.5M") == 1_500_000

    def test_whitespace(self):
        assert clean_money("  €25.00m  ") == 25_000_000

    def test_nan_passthrough(self):
        assert clean_money(float("nan")) is None

    def test_invalid_returns_none(self):
        assert clean_money("not a price") is None

    def test_zero(self):
        assert clean_money("€0") == 0


class TestNormalizeName:
    def test_strips_diacritics(self):
        assert normalize_name("Vinícius Júnior") == "vinicius junior"

    def test_lowercases(self):
        assert normalize_name("Erling HAALAND") == "erling haaland"

    def test_collapses_whitespace(self):
        assert normalize_name("  Phil   Foden  ") == "phil foden"

    def test_strips_punctuation(self):
        assert normalize_name("N'Golo Kanté") == "ngolo kante"

    def test_nan_returns_none(self):
        assert normalize_name(float("nan")) is None

    def test_matches_diacritic_variants(self):
        """Two spellings of the same player must produce the same key."""
        assert normalize_name("Hakan Çalhanoğlu") == normalize_name("Hakan Calhanoglu")


class TestPipelineSanity:
    """Smoke test on the actual processed dataset, if present."""

    @pytest.fixture
    def dataset(self):
        path = Path(__file__).resolve().parents[1] / "data" / "processed" / "final_dataset.csv"
        if not path.exists():
            pytest.skip("final_dataset.csv not present — run src/data_cleaning.py first")
        return pd.read_csv(path)

    def test_no_negative_values(self, dataset):
        assert (dataset["value_eur"] >= 0).all()

    def test_unique_name_keys(self, dataset):
        assert dataset["name_key"].is_unique

    def test_age_is_realistic(self, dataset):
        ages = dataset["age"].astype(str).str.split("-").str[0].astype(int)
        assert ages.between(15, 45).all()
