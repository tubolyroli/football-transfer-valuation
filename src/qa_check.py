import pandas as pd
import great_expectations as gx

# 1. Simulate "Dirty" Scraped Data
data = {
    'player_id': [101, 102, 103, 101, 105],  # Error: Duplicate ID (101)
    'player_name': ['Szoboszlai Dominik', 'Phil Foden', 'Harry Kane', 'Szoboszlai Dominik', 'Unknown Player'],
    'age': [24, 23, 30, 24, 150],            # Error: Age 150 is impossible
    'market_value_eur': [75000000, 110000000, 90000000, 75000000, -5000], # Error: Negative value
    'league': ['Premier League', 'Premier League', 'Bundesliga', 'Premier League', None] # Error: Null league
}

# Create DataFrame (simulate loading from CSV)
df = pd.DataFrame(data)

print("--- RAW DATA SNAPSHOT ---")
print(df)
print("\n--- STARTING VALIDATION ---")

# 2. Initialize Great Expectations Context
context = gx.get_context()

# 3. Create a Validator Object linked to the DataFrame
datasource = context.sources.add_pandas(name="football_data_source")
asset = datasource.add_dataframe_asset(name="players", dataframe=df)
batch_request = asset.build_batch_request()
validator = context.get_validator(batch_request=batch_request, expectation_suite_name="transfer_market_checks")

# 4. Define Your "Expectations" (The Rules)

# Rule A: Player IDs must be unique
validator.expect_column_values_to_be_unique(column="player_id")

# Rule B: Age must be realistic (e.g., between 15 and 45)
validator.expect_column_values_to_be_between(column="age", min_value=15, max_value=45)

# Rule C: Market Value cannot be missing and must be positive
validator.expect_column_values_to_not_be_null(column="market_value_eur")
validator.expect_column_values_to_be_between(column="market_value_eur", min_value=0, max_value=300000000)

# Rule D: League must be one of the specific set we track
validator.expect_column_values_to_be_in_set(
    column="league", 
    value_set=["Premier League", "Bundesliga", "La Liga", "Serie A", "Ligue 1", "NB I"]
)

# 5. Run Validation and Save Results
results = validator.validate()

# 6. Output for the User
if not results["success"]:
    print("\n[CRITICAL] Data Quality Checks FAILED!")
    print(f"Success Score: {results['statistics']['success_percent']}%")
    # Loop through results to show exactly what failed
    for res in results["results"]:
        if not res["success"]:
            print(f"FAILED CHECK: {res['expectation_config']['expectation_type']} on column {res['expectation_config']['kwargs']['column']}")
            print(f"   Details: Found {res['result']['unexpected_count']} errors.")
else:
    print("\n[SUCCESS] All Data Quality Checks Passed. Ready for Pipeline.")
