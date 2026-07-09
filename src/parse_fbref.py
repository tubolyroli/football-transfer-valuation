"""Parse the cached FBref 'Big-5 European Leagues' standard stats HTML into a CSV.

I deliberately work from a *manually-downloaded* HTML snapshot stored at
`data/raw/fbref_page.html` rather than scraping live, because:

1. FBref's anti-bot protection (Cloudflare) reliably blocks `requests` and the
   throttling needed to scrape politely is slower than just downloading the
   page once a season from a browser.
2. The HTML page is treated as the cached raw artifact — the parse step is
   deterministic and reproducible across runs, which is what matters for the
   downstream pipeline.

To refresh the data:
    1. Open https://fbref.com/en/comps/Big5/stats/players/Big-5-European-Leagues-Stats in a browser.
    2. Save the page as `data/raw/fbref_page.html`.
    3. Re-run this script.
"""

import os
from io import StringIO

import pandas as pd

INPUT_HTML = "data/raw/fbref_page.html"
OUTPUT_CSV = "data/raw/fbref_stats.csv"


def parse_local_html() -> pd.DataFrame | None:
    print(f"[INFO] Reading cached FBref HTML from {INPUT_HTML}...")
    if not os.path.exists(INPUT_HTML):
        print(f"[ERROR] {INPUT_HTML} not found. See module docstring for refresh instructions.")
        return None

    try:
        with open(INPUT_HTML, "r", encoding="utf-8") as f:
            html_content = f.read()

        tables = pd.read_html(StringIO(html_content), attrs={"id": "stats_standard"})
        df = tables[0]

        df.columns = ["_".join(col).strip() for col in df.columns.values]
        df = df.rename(columns={
            "Unnamed: 0_level_0_Rk": "rank",
            "Unnamed: 1_level_0_Player": "player_name",
            "Unnamed: 2_level_0_Nation": "nation",
            "Unnamed: 3_level_0_Pos": "position",
            "Unnamed: 4_level_0_Squad": "team",
            "Unnamed: 5_level_0_Comp": "league",
            "Unnamed: 6_level_0_Age": "age",
            "Performance_Gls": "goals",
            "Performance_Ast": "assists",
            "Expected_xG": "xg",
        })

        df = df[df["player_name"] != "Player"]
        df = df.drop_duplicates(subset=["player_name", "team"])

        print(f"[INFO] Parsed {len(df)} players from cached HTML.")
        return df

    except Exception as e:
        print(f"[ERROR] Failed to parse cached FBref HTML: {e}")
        return None


if __name__ == "__main__":
    df_stats = parse_local_html()
    if df_stats is not None:
        df_stats.to_csv(OUTPUT_CSV, index=False)
        print(f"[SUCCESS] Stats saved to {OUTPUT_CSV}")
