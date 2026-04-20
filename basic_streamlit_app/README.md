# V-DEM Core Indices Visualization Tool 

An interactive Streamlit dashboard for exploring democracy data from the [Varieties of Democracy (V-Dem) Dataset](https://www.v-dem.net/). Users can select a country and time range of interest to visualize five core democracy indicies as line graphs with descriptions of each index displayed above its plot.

## Features

- **Country Selector**: Choose any country included in the dataset from the dropdown menu
- **Year Range Slider**: Filter the data to a custom time period
- **5 Core Democracy Indicies**: Each index is displayed as its own labeled line graph with a short description of what it measures

## Dataset

"The Varieties of Democracy (V-Dem) Research Project takes a comprehensive approach to understanding democratization. This approach encompasses multiple core principles: electoral, liberal, majoritarian, consensual, participatory, deliberative, and egalitarian. Each Principle is represented by a separate index, and each is regarded as a separate outcome in the proposed study. In this manner we reconceptualize democracy from a single outcome to a set of outcomes.

In addition, we break down each core principle into its constituent components, each to be measured separately. Components include features such as free and fair elections, civil liberties, judicial independence, executive constraints, gender equality, media freedom, and civil society. Finally, each component is disaggregated into specific indicators.

This fundamentally different approach to democratization is made possible by the V-Dem Database, which measures 600+ indicators annually from 1789 to the present for all countries of the world.

The V-Dem approach stands out, first, as a large global collaboration among scholars with diverse areas of expertise; second, as the first project attempting to explain different varieties of democracy; and third, thanks to the highly disaggregated V-Dem data, the first project to explore causal mechanisms linking different aspects of democracy together." - [The V-Dem Institute](https://www.v-dem.net/about/v-dem-project/)

## Dependencies
- [Streamlit](https://streamlit.io): Web app framework
- [Pandas](https://pandas.pydata.org/): Data loading and filtering

## How to Run the App
**1. Clone the repository**
```bash
git clone https://github.com/millerjacobt/Miller-Data-Science-Portfolio.git
cd Miller-Data-Science-Portfolio
```

**2. Install dependencies**
```bash
pip install streamlit pandas
```

**3. Launch the Streamlit app from the root of the repository**
```bash
streamlit run basic_streamlit_app/main.py
```

## About

This project was built as a part of a data science course to demonstrate interactive data visualization using Streamlit and Python. It highlights the use of Pandas for data wrangling and Streamlit for the production of this user friendly data visualization app. 
