# Football Transfer Market Valuation

An end-to-end data science pipeline that estimates the "fair market value" of European
footballers by combining performance stats (FBref) with market valuations
(Transfermarkt), then surfaces under-/over-valued players via an interactive dashboard.

## What this project demonstrates
* **Data engineering:** Multi-source ingestion, fuzzy name matching with diacritic
  normalization (Vinícius Júnior ↔ Vinicius Junior), reproducible pipeline.
* **Data quality:** Automated Great Expectations checks that gate the pipeline.
* **Modeling:** Ridge regression on log-transformed prices with categorical encoding;
  three model leaderboard (Ridge / RandomForest / GradientBoosting) with held-out and
  5-fold cross-validated R².
* **Analysis:** Quantifies the league-level price premium ("Premier League Tax") via
  the model's standardized coefficients.
* **Communication:** EDA notebook with the story, plus a four-tab Streamlit dashboard
  (predictions, residuals, league premium, feature importance).
* **Testing:** 17 pytest tests on the cleaning utilities + pipeline output.

## Results (top-500 most-valuable players, 405 matched)
| Model            | Held-out R² | Held-out MAE | 5-fold CV R² (log) |
|------------------|-------------|--------------|--------------------|
| **Ridge**        | **0.25**    | **€14.9M**   | **0.16 ± 0.03**    |
| RandomForest     | 0.05        | €15.4M       | 0.09 ± 0.05        |
| GradientBoosting | -0.05       | €16.1M       | 0.08 ± 0.04        |

Ridge wins on this small, high-noise dataset — exactly the kind of regime where
high-variance tree ensembles overfit.

## Project Structure
```
src/
  parse_fbref.py           Parse cached FBref HTML → CSV
  transfermarkt_scraper.py Scrape Transfermarkt market values
  data_cleaning.py         Normalize names, merge, write final dataset
  qa_check.py              Great Expectations validation suite
  model.py                 Train & compare three models
  app.py                   Streamlit dashboard (4 tabs)
notebooks/
  01_eda.ipynb             Exploratory analysis & modeling decisions
tests/
  test_data_cleaning.py    17 tests on cleaning + pipeline sanity
data/
  raw/                     Source CSVs + cached FBref HTML snapshot
  processed/               Merged master table
```

## Setup & Reproducibility
```bash
pip install -r requirements.txt

# 1. Build the dataset
python src/parse_fbref.py             # parse cached FBref HTML
python src/transfermarkt_scraper.py   # scrape Transfermarkt
python src/data_cleaning.py           # merge + normalize names

# 2. Validate data quality (exits non-zero on failure — CI-ready)
python src/qa_check.py

# 3. Train & evaluate models
python src/model.py

# 4. Launch the dashboard
streamlit run src/app.py

# 5. Run tests
pytest tests/ -v
```

## Tech Stack
Python 3.10+ · pandas · scikit-learn · Great Expectations · Streamlit · Plotly · pytest

## Design Choices Worth Noting
* **Cached FBref HTML.** FBref aggressively blocks scrapers; the raw HTML is treated as
  a versioned input artifact rather than scraped live. See `src/parse_fbref.py` for
  refresh instructions.
* **Log-transformed target.** Raw market value has skew ≈ 2.6 (Haaland tail); log1p
  reduces skew to ≈ 1.1 and gives a ~40% relative R² improvement.
* **Honest out-of-sample reporting.** All R²/MAE values come from a held-out 20% test
  set, not in-sample fits. "Undervalued" rankings are computed on test rows only.
* **Selection bias is acknowledged.** Sample is the top-500 most-valuable players, so
  the model learns to discriminate *among* expensive players — not to identify expensive
  players from nothing.

## Limitations
* No contract length, no commercial/marketing value, no injury history.
* Single season of stats (a 3-season window would surface true outliers, not random ones).
* Position is collapsed to primary role; multi-position flexibility is ignored.
