import requests
import pandas as pd
import time
import random
import os

def scrape_transfermarkt(pages=20):
    """
    Scrapes the top 500 most valuable players from Transfermarkt.
    pages: Number of pages to scrape (25 players per page).
    """
    base_url = "https://www.transfermarkt.com/spieler-statistik/wertvollstespieler/marktwertetop/plus/?page={}"
    
    # Headers must be very specific for Transfermarkt
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Referer": "https://www.google.com/"
    }

    all_players = []
    
    print(f"[INFO] Starting Transfermarkt scrape for {pages} pages...")

    for page in range(1, pages + 1):
        url = base_url.format(page)
        print(f"[INFO] Scraping page {page}/{pages}...")
        
        try:
            # Random delay to look human (2-5 seconds)
            time.sleep(random.uniform(2, 5))
            
            response = requests.get(url, headers=headers)
            
            if response.status_code == 403:
                print(f"[CRITICAL] Blocked by Transfermarkt on page {page}. Stopping early.")
                break
                
            response.raise_for_status()
            
            # Use Pandas to parse the HTML table
            # Transfermarkt tables are complex; we target the main one
            dfs = pd.read_html(response.text)
            
            # usually the main table is the one with 'Market value' in columns
            target_df = None
            for df in dfs:
                if 'Market value' in df.columns or 'Value' in df.columns:
                    target_df = df
                    break
            
            if target_df is None:
                # Fallback: The main table is usually index 1 on this specific page type
                target_df = dfs[1]

            # Clean and append
            # We only need Name and Value (and maybe Age/Team for matching)
            # Transfermarkt structure often puts Name in column 3 or 'Player'
            # This part is fragile and depends on TM's current layout
            
            # Let's save the raw DF for now to clean later
            all_players.append(target_df)

        except Exception as e:
            print(f"[ERROR] Failed on page {page}: {e}")

    if not all_players:
        return None

    # Combine all pages
    final_df = pd.concat(all_players, ignore_index=True)
    print(f"[SUCCESS] Scraped {len(final_df)} rows.")
    return final_df

if __name__ == "__main__":
    os.makedirs("data/raw", exist_ok=True)
    
    # Run Scraper
    df_val = scrape_transfermarkt(pages=20) # 20 pages = 500 players
    
    if df_val is not None:
        output_path = "data/raw/transfermarkt_values.csv"
        df_val.to_csv(output_path, index=False)
        print(f"[SUCCESS] Data saved to {output_path}")
