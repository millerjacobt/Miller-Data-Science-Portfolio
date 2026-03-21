# Olympic Medals Analysis Project

## Project Overview
This project explores Olympic medal data using Python and demonstrates **tidy data principles**. The goal is to clean and transform the dataset into a tidy format where:

- Each variable forms a column  
- Each observation forms a row  
- Each type of observational unit forms a table  

From there, I generate pivot tables and visualizations to analyze medal counts by **sport, gender, and athlete**

---

## Instructions

### 1. Dependencies
Make sure you have the following Python packages installed via **conda**:
(Install anaconda on your machine if you do not already have it)

- **pandas** 
(For data cleaning & manipulation)
- **matplotlib**
(For data visualization)

You can install them with:

```bash
conda install pandas matplotlib 
```

### 2. Running the Notebook
- Download the 'tidy_data.ipynb' notebook and 'olympics_08_medalists.csv' dataset
- Inside vs_code, open the tidy_data notebook and adjust the filepath to reflect the olympics dataset's location on your machine
- Run the cells from top to bottom and feel free to augment or add to the code!

## Dataset Description
This dataset includes the medal winners from the 2008 Summer Olympics. It includes name, medal, sport/event, and gender. It is coded in a wide format, so it needed significant transformations to be more readable, tidy data.

## Resources 
For further information on Pandas and tidy data as a concept, please refer to [Pandas Cheat Sheet] (https://pandas.pydata.org/Pandas_Cheat_Sheet.pdf) and [The Tidy Data Paper] (https://vita.had.co.nz/papers/tidy-data.pdf)

## Visualizations 
In my analysis, I created some simple visualizations from the limited sets of variables available. My favorite was my bar graph showing the distribution of medals by sport and gender, shown below:

[Medal Distribution by Sport and Gender] ('/Users/work/Documents/GitHub/Miller-Data-Science-Portfolio/Images/Gender Distribution by Sport.png')