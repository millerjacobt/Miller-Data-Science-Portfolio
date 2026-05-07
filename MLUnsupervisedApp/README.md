# Hidden Structures: An Interactive Unsupervised ML App

This app allows users to explore three unsupervised machine learning techniques: Principal Component Analysis (PCA), K-Means Clustering, and Hierarchical Clustering. Users can upload a CSV or choose from two sample datasets, perform basic preprocessing, select feature variables, and experiment with hyperparameters to discover hidden structure in their data. Visualizations and explanations are provided throughout to help users interpret their results.

## Live App
[Launch App](https://your-app-url.streamlit.app)

## How to Run Locally
### Prerequisites
- Python 3.8+
- streamlit
- pandas
- numpy
- scikit-learn
- scipy
- matplotlib
- seaborn

### Installation
1. Clone the repository:
```bash
git clone https://github.com/millerjacobt/Miller-Data-Science-Portfolio.git
```
2. Navigate to the MLUnsupervisedApp folder:
```bash
cd Miller-Data-Science-Portfolio/MLUnsupervisedApp
```
3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Run the app:
```bash
streamlit run app.py
```

## Sample Datasets
### Breast Cancer
The Breast Cancer dataset was originally collected by Dr. William H. Wolberg at the University of Wisconsin Hospitals. The dataset includes 569 patient records with measurements of cell nuclei from digitized images of breast mass biopsies (features such as radius, texture, smoothness, etc). It is featured as a built-in benchmark dataset in the scikit-learn package and is available via ```sklearn.datasets.load_breast_cancer```

### Titanic
The Titanic dataset is a widely used demo dataset available on Kaggle originally collected by the British Board of Trade after the Titanic sank in 1912. It includes 891 passenger records with features such as passenger class (Pclass), age, sex, fare paid, etc. This is an interesting dataset to explore as it includes social determinants such as socioeconomic class, gender, and age.

## Techniques
### Principal Component Analysis (PCA)
PCA is a dimensionality reduction technique that compresses many features into a smaller set of components, each capturing a portion of the total variance in the data. It is useful for visualizing high-dimensional data and identifying the most significant variance patterns.

#### Hyperparameters
- **Number of Components:** The number of principal components to compute. Choosing 2 or 3 components allows for easy visualization, while more components capture more variance but may be harder to interpret.

### K-Means Clustering
K-Means partitions the data into k clusters by minimizing the within-cluster sum of squared distances from each point to its assigned cluster centroid. It is used to create groups of similar data points based on their features.

#### Hyperparameters
- **Number of Clusters (k):** The number of clusters the model will create. Use the elbow plot and silhouette score after running the model to determine the best k value.
- **Max Iterations:** The maximum number of iterations the algorithm will run. Increasing this value can help ensure convergence but may increase computation time.

### Hierarchical Clustering
Hierarchical Clustering builds a tree of clusters by successively merging the most similar points together. The result is a dendrogram showing the full merge history. By cutting the tree at a chosen level, you can select the number of clusters that best fits your data.

#### Hyperparameters
- **Number of Clusters (k):** The number of clusters to form by cutting the dendrogram. Use the dendrogram and silhouette score plot to determine the best k value.
- **Linkage Method:** Controls how distance between clusters is measured during merging. 'Ward' minimizes within-cluster variance and is the best default. 'Complete', 'average', and 'single' use different distance strategies.

## References
- [scikit-learn KMeans Documentation](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [scikit-learn AgglomerativeClustering Documentation](https://scikit-learn.org/stable/modules/clustering.html#hierarchical-clustering)
- [scikit-learn PCA Documentation](https://scikit-learn.org/stable/modules/decomposition.html#pca)
- [scipy Dendrogram Documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html)
- [Breast Cancer Wisconsin Dataset - sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html)
- [Titanic Dataset - Kaggle](https://www.kaggle.com/competitions/titanic)
- [Streamlit Documentation](https://docs.streamlit.io)

## Author
**Jacob Miller**
University of Notre Dame, Class of 2026
[GitHub](https://github.com/millerjacobt) | [LinkedIn](https://www.linkedin.com/in/millerjacobt/)