# Branches or Neighborhoods: An Interactive Supervised ML App

This is an app that allows users to explore the Decision Tree and K Nearest Neighbor supervised machine learning models. Users are encouraged to spend time changing their feature variables and adjusting the hyperparameters of the models to find the best model for their dataset. Users can upload a csv or choose from three interesting sample datasets, perform basic preprocessing, choose their target and feature variables, then run and analyse their unique models. By hovering over the "?" next to the outcome metrics and selecting the dropdown tabs above the matrices and graph, users can learn more about each metric to understand how their result relates to their model. This app allows users to quickly test different models to categorize their data, comparing the "branches" and "neighbors" of the Decision Tree and K Nearest Neighbor ML Models.

## Live App
[Launch App](https://branches-or-neighborhoods.streamlit.app)

## How to Run Locally
### Prerequisites
- Python 3.8+
- streamlit
- pandas
- scikit-learn
- matplotlib
- seaborn

### Installation
1. Clone the repository:
```bash
git clone https://github.com/millerjacobt/Miller-Data-Science-Portfolio.git
```
2. Navigate to the MLStreamlitApp folder
```bash
cd Miller-Data-Science-Portfolio/MLStreamlitApp
```
3. Install Dependencies:
```bash
pip install -r requirements.txt
```
4. Run the app:
```bash
streamlit run app.py
```

## Sample Datasets
### BRFSS Diabetes Health Indicators
The Behavioral Risk Factor Surveillance System (BRFSS) is an annual CDC telephone survey collecting health and lifestyle data from U.S. Residents. This 2015 collection contains 70,692 survey responses with 21 social determinants of health features such as income, education, physical activity, and BMI, with a binary target variable indicating whether a respondent has diabetes or prediabetes. This specific dataset was cleaned and balanced by Alex Teboul to have an equal 50-50 split of respondents with no diabetes and with either prediabetes or diabetes. The typical target variable `Diabetes_binary` has 2 classes. 0 is for no diabetes, and 1 is for prediabetes or diabetes. Users can access the cleaned, balanced dataset on [Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset). 
Original data was sourced from the [CDC BRFSS 2015 Survey](https://www.cdc.gov/brfss).

### Breast Cancer
The Breast Cancer dataset was originally collected by Dr. William H. Wolberg at the University of Wisconsin Hospitals. The dataset includes 569 patient records with measurements of cell nuclei from digitized images of breast mass biopsies (features such as radius, texture, smoothness, etc). The typical target variable `target` is coded as 0 for a malignant tumor and 1 for a benign. It is featured as a built in benchmark dataset in the scikit learn package and is available via ```sklearn.datasets.load_breast_cancer```

### Titanic
The Titanic dataset is a widely used demo dataset available on Kaggle originally collected by the British Board of Trade after the Titanic sank in 1912. It includes 891 passenger records with features such as passenger class (Pclass), age, sex, fare paid, etc. The typical target variable `Survived` is coded as 0 = did not survive, 1 = survived. This is an interesting dataset to model as it includes social determinants such as socioeconomic class, gender, and age, allowing users to test theories regarding these features. 

## Models
### Decision Tree (DT)
A decision tree is a classification model that groups cases by following a series of yes/no questions down the "tree" from the root node to the leaf nodes. Decision nodes function as filters, sorting each case until it reaches a leaf node that assigns a final classification. Decision trees are beneficial because they are interpretable (they mirror human decision making), work without scaling the data, and are fast to train.

#### Hyperparameters
- **Max Depth (DT):**
Max Depth is the number of levels a DT model uses. The higher the max depth, the larger the tree. More levels allow for more branches and nodes, but be careful when adding levels as too many can lead to an overfitted model to the training data and lead to inaccurate predictions.
- **Min Samples Split (DT):**
Minimum Samples Split is the sample size needed for the DT model to make another split. A higher min samples split will limit the model from creating rules for small groups of data: potentially limiting overfitting the data.
- **Criterion (DT):**
Criterion is the function used to measure split quality at each node. `gini` measures disorder in each split while `entropy` measures diversity using probability. Results are often similar: try both and compare!

### K Nearest Neighbor (KNN)
K-Nearest Neighbors (KNN) is a classification model that represents data as points in multidimensional space and calculates similarity based on the physical distance between points. To classify a new point, the model finds the k nearest neighbors and assigns the majority class among them as the prediction. KNN models are particularly good at capturing more complex, non-linear relationships as it considers all features simultaneously through distance rather than splitting on one feature at a time.

#### Hyperparameters
- **Number of Neighbors / K (KNN):**
The Number of Nearest Neighbors to use for classification. A smaller K can lead to a more flexible model that may capture more complex patterns but may also be sensitive to noise in the data, while a larger K can lead to a smoother decision boundary that may generalize better but may also miss important patterns.
- **Weights (KNN):** The weight function used in prediction. `uniform` treats all neighbors equally while `distance` weights closer neighbors more heavily, giving them greater influence on the final prediction.

## References
- [scikit-learn Decision Tree Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [scikit-learn KNN Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html)
- [BRFSS Diabetes Dataset - Alex Teboul on Kaggle](https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset)
- [CDC BRFSS 2015 Survey](https://www.cdc.gov/brfss)
- [Breast Cancer Wisconsin Dataset - sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- [Titanic Dataset - Kaggle](https://www.kaggle.com/competitions/titanic)
- [Streamlit Documentation](https://docs.streamlit.io)

## Author
**Jacob Miller**
University of Notre Dame, Class of 2026
[GitHub](https://github.com/millerjacobt) | [LinkedIn](https://www.linkedin.com/in/millerjacobt/)