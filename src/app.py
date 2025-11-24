import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# --- Page Config ---
st.set_page_config(page_title="Football Transfer Valuation", page_icon="⚽", layout="wide")

# --- 1. Load & Cache Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/final_dataset.csv")
    # Clean Age
    df['age_clean'] = df['age'].str.split('-').str[0].astype(int)
    return df

df = load_data()

# --- 2. Sidebar Filters ---
st.sidebar.header("Filter Players")
selected_leagues = st.sidebar.multiselect(
    "Select Leagues", 
    options=df['league'].unique(), 
    default=df['league'].unique()
)
min_minutes = st.sidebar.slider("Minimum Minutes Played", 0, 3000, 500)

# Filter the dataframe
df_filtered = df[
    (df['league'].isin(selected_leagues)) & 
    (df['Playing Time_Min'] >= min_minutes)
]

# --- 3. Train Model (On the Fly) ---
features = ['age_clean', 'goals', 'assists', 'xg', 'Progression_PrgC']
target = 'value_eur'

X = df_filtered[features]
y = df_filtered[target]

# Train
model = Ridge(alpha=1.0)
model.fit(X, y)
df_filtered['predicted_value'] = model.predict(X)
df_filtered['difference'] = df_filtered['predicted_value'] - df_filtered['value_eur']

# Metrics
r2 = model.score(X, y)

# --- 4. Dashboard Layout ---
st.title("⚽ Football Transfer Valuation Model")
st.markdown("""
This tool uses **Machine Learning (Ridge Regression)** to estimate the 'Fair Market Value' of players based on their performance stats (Goals, xG, Progression).
Any positive difference suggests the player might be **Undervalued** (a bargain).
""")

# Top Level Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Players Analyzed", len(df_filtered))
col2.metric("Model Accuracy (R²)", f"{r2:.2%}")
col3.metric("Market Inefficiency Found", f"€{df_filtered['difference'].sum():,.0f}")

# --- 5. Visualizations ---
st.subheader("💰 Actual Market Value vs. Model Prediction")
fig = px.scatter(
    df_filtered, 
    x='value_eur', 
    y='predicted_value', 
    hover_data=['player_name', 'age', 'team'],
    color='difference',
    color_continuous_scale='RdBu',
    labels={'value_eur': 'Actual Market Value (€)', 'predicted_value': 'Model Predicted Value (€)'},
    title="Players above the diagonal are Undervalued (Blue)"
)
# Add a diagonal line (Perfect prediction)
fig.add_shape(type="line", x0=0, y0=0, x1=150000000, y1=150000000, line=dict(color="Green", dash="dash"))
st.plotly_chart(fig, use_container_width=True)

# --- 6. The "Bargain Bin" Table ---
st.subheader("💎 Top 10 Undervalued Players")
undervalued = df_filtered.sort_values(by='difference', ascending=False).head(10)
st.dataframe(
    undervalued[['player_name', 'age', 'team', 'value_eur', 'predicted_value', 'difference']]
    .style.format({'value_eur': '€{:,.0f}', 'predicted_value': '€{:,.0f}', 'difference': '€{:,.0f}'})
)

# --- 7. Explainability ---
st.markdown("### 📝 Methodology")
st.markdown(f"""
* **Data Source:** FBref (Performance) & Transfermarkt (Price).
* **Features Used:** Age, Goals, Assists, Expected Goals (xG), Progressive Carries.
* **Limitations:** The model currently does not account for contract length or commercial value (e.g., shirt sales).
""")
