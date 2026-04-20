# V-DEM Core Indices Visualization Tool 

An interactive Streamlit dashboard for exploring democracy data from the [Varieties of Democracy (V-Dem) Dataset](https://www.v-dem.net/). Users can select one or multiple countries and a time range of interest, visualize five core democracy indicies (descriptions of each index displayed), and compare the levels of each index over time and between countries.

## Features

- **Country Selector**: Select any countries included in the dataset from the dropdown menu
- **Year Range Slider**: Filter the data to a custom time period
- **5 Core Democracy Indices**: Each index is displayed as its own labeled line graph with a short description of what it measures

## Dataset
This project uses the V-Dem dataset to allow users to visualize differences in democratic components over time and between countries. A short blurb about the V-Dem project is shown below:

> [!NOTE]
> "The Varieties of Democracy (V-Dem) Research Project takes a comprehensive approach to understanding democratization. This approach encompasses
> multiple core principles: electoral, liberal, majoritarian, consensual, participatory, deliberative, and egalitarian. Each Principle is represented by
> a separate index, and each is regarded as a separate outcome in the proposed study. In this manner we reconceptualize democracy from a single outcome
> to a set of outcomes.
>
> The V-Dem approach stands out, first, as a large global collaboration among scholars with diverse areas of expertise; second, as the first project
> attempting to explain different varieties of democracy; and third, thanks to the highly disaggregated V-Dem data, the first project to explore causal
> mechanisms linking different aspects of democracy together."
>   - [The V-Dem Institute](https://www.v-dem.net/about/v-dem-project/)

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
