import pandas as pd
import re
import os
from unidecode import unidecode


def clean_money(value_str):
    """Converts '€200.00m' to 200000000. Handles 'm' (millions) and 'k' (thousands)."""
    if pd.isna(value_str):
        return None

    value_str = str(value_str).lower().replace('€', '').strip()

    multiplier = 1
    if 'm' in value_str:
        multiplier = 1_000_000
        value_str = value_str.replace('m', '')
    elif 'k' in value_str:
        multiplier = 1_000
        value_str = value_str.replace('k', '')

    try:
        return float(value_str) * multiplier
    except ValueError:
        return None


def normalize_name(name):
    """Normalize player names for fuzzy joining: strip diacritics, lowercase, collapse whitespace."""
    if pd.isna(name):
        return None
    s = unidecode(str(name)).lower().strip()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def process_data():
    print("[INFO] Loading raw datasets...")
    df_fb = pd.read_csv("data/raw/fbref_stats.csv")
    df_tm = pd.read_csv("data/raw/transfermarkt_values.csv")

    print(f"[INFO] Cleaning Transfermarkt data ({len(df_tm)} raw rows)...")

    # The clean name lives in the row BELOW the value row; shift it up to align.
    df_tm['clean_name'] = df_tm['Player'].shift(-1)
    df_tm = df_tm.dropna(subset=['Market value'])
    df_tm['value_eur'] = df_tm['Market value'].apply(clean_money)

    df_tm = df_tm[['clean_name', 'value_eur', 'Age']].rename(
        columns={'clean_name': 'player_name', 'Age': 'age_tm'}
    )
    df_tm['name_key'] = df_tm['player_name'].apply(normalize_name)
    df_tm = df_tm.dropna(subset=['name_key'])
    df_tm = df_tm[df_tm['name_key'] != '']
    print(f"[INFO] Transfermarkt clean: {len(df_tm)} players found.")

    print(f"[INFO] Cleaning FBref data ({len(df_fb)} rows)...")
    df_fb['name_key'] = df_fb['player_name'].apply(normalize_name)

    print("[INFO] Merging datasets on normalized name key...")
    merged_df = pd.merge(
        df_fb,
        df_tm.drop(columns=['player_name']),
        on='name_key',
        how='inner',
    )
    merged_df = merged_df.drop_duplicates(subset=['name_key'])

    matched = len(merged_df)
    print(f"[SUCCESS] Final Dataset: {matched} players ready for modeling.")
    print(f"[INFO] Match rate: {matched}/{len(df_tm)} TM players ({matched/len(df_tm):.1%})")

    return merged_df


if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)

    final_df = process_data()

    output_path = "data/processed/final_dataset.csv"
    final_df.to_csv(output_path, index=False)
    print(f"[SAVED] Master table saved to {output_path}")
