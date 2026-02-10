import streamlit as st
import pandas as pd

st.title("My First EDA App")

# Interactive button
if st.button("Click me!"):
    st.write("🎉 You clicked the button!")

# Color picker
color = st.color_picker("Pick a color", "#00f900")
st.write(f"You picked: {color}")

# Load data from CSV
st.subheader("Data Explorer")
df = pd.read_csv("sample_data-1.csv")

# Show full dataset
st.write("Full dataset:")
st.dataframe(df)

# City filter
city = st.selectbox("Select a city", df["City"].unique())
filtered_df = df[df["City"] == city]

st.write(f"People in {city}:")
st.dataframe(filtered_df)

## Add bar chart
st.bar_chart(df["Salary"])

import seaborn as sns

box_plot1 = sns.boxplot(x=df["City"], y=df["Salary"])

st.pyplot(box_plot1)


