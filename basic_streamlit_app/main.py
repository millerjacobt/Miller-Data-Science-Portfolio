import streamlit as st
import pandas as pd

# Title
st.title("V-Dem Democracy Indices Explorer")

# Description
st.markdown(
    """
    This app explores five core democracy indices from the
    Varieties of Democracy (V-Dem) dataset.
    
    Use the controls in the sidebar to filter by country and year.
    """
)

# Load data
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

df = pd.read_csv(BASE_DIR / "data" / "vdem_subset.csv")

# Sidebar controls
st.sidebar.markdown("## Filters")

countries = st.sidebar.multiselect(
    "Select countries",
    sorted(df["country_name"].unique()),
    default=["United States of America"]
)

year_range = st.sidebar.slider(
    "Select year range",
    int(df["year"].min()),
    int(df["year"].max()),
    (1990, int(df["year"].max()))
)

# Filter data
filtered_df = df[
    (df["country_name"].isin(countries)) &
    (df["year"] >= year_range[0]) &
    (df["year"] <= year_range[1])
]

# Select the 5 indices
indices = [
    "v2x_libdem",
    "v2x_polyarchy",
    "v2x_partipdem",
    "v2x_delibdem",
    "v2x_egaldem"
]

# Store descriptions for each index
index_descriptions = {
    "v2x_libdem": "Liberal Democracy Index: captures the extent to which individual and minority rights are protected and government power is constrained.",
    "v2x_polyarchy": "Electoral Democracy (Polyarchy): measures the extent to which leaders are chosen through free and fair elections.",
    "v2x_partipdem": "Participatory Democracy: reflects the degree to which citizens actively participate in political processes.",
    "v2x_delibdem": "Deliberative Democracy: measures the extent to which political decisions are made through reasoned, inclusive deliberation.",
    "v2x_egaldem": "Egalitarian Democracy: captures the extent to which political power is distributed equally across social groups."
}

table_columns = ["country_name"] + indices

# Plot each index with description attached
if not countries:
    st.warning("Please select at least one country.")
else:
    for idx in indices:
        st.markdown(f"### {idx}")
        st.markdown(index_descriptions[idx])
        chart_df = filtered_df.pivot(index="year", columns="country_name", values=idx)
        st.line_chart(chart_df)
    

# Show data table
st.subheader("Democracy Indices (Filtered)")
st.dataframe(filtered_df[["year"] + table_columns].sort_values(["country_name", "year"]))
