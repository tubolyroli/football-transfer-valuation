"""Compact euro formatting shared by the dashboard."""


def fmt_eur(value) -> str:
    """Format a euro amount compactly: €7.8M, €500K, €250.

    Trailing '.0' on millions is dropped (€45M, not €45.0M). NaN/None → em dash.
    """
    if value is None or value != value:
        return "—"
    sign = "-" if value < 0 else ""
    v = abs(value)
    if v >= 1_000_000:
        num = f"{v / 1_000_000:.1f}".removesuffix(".0")
        return f"{sign}€{num}M"
    if v >= 1_000:
        num = f"{v / 1_000:.1f}".removesuffix(".0")
        return f"{sign}€{num}K"
    return f"{sign}€{v:,.0f}"
