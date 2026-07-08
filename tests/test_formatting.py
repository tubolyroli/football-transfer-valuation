import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.formatting import fmt_eur


class TestFmtEur:
    def test_millions_one_decimal(self):
        assert fmt_eur(7_764_536) == "€7.8M"

    def test_millions_strips_trailing_zero(self):
        assert fmt_eur(45_000_000) == "€45M"

    def test_thousands(self):
        assert fmt_eur(500_000) == "€500K"

    def test_small_value(self):
        assert fmt_eur(250) == "€250"

    def test_negative(self):
        assert fmt_eur(-1_200_000) == "-€1.2M"

    def test_zero(self):
        assert fmt_eur(0) == "€0"

    def test_nan(self):
        assert fmt_eur(float("nan")) == "—"

    def test_none(self):
        assert fmt_eur(None) == "—"
