# Import necessary libraries and tools for data manipulation, machine learning, and visualization
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage


# Setup Streamlit app title and description
st.set_page_config(page_title="An Interactive Unsupervised ML App", layout="wide")

st.title("An Interactive Unsupervised Machine Learning App")
st.write("This app allows you to explore and apply unsupervised machine learning techniques, including Principal Component Analysis (PCA), K-Means Clustering, and Hierarchical Clustering. You can upload your own dataset or use a sample dataset to experiment with different algorithms and hyperparameters. The app provides visualizations and explanations to help you understand the results of each technique.")

# Create a function to preprocess the future data: Encode categorical values, handle missing values, and drop variables with high cardinality (variables with more than 20 unique values)
def preprocess_data(df, missing_threshold, fill_method):
    # Use a list comprehension to identify columns where more than missing_threshold% of the values are missing
    high_mi_vars = [col for col in df.columns if df[col].isnull().mean() > missing_threshold]

    # Identify text columns with more than 20 unique values (high cardinality) (names, ticket numbers, etc.)
    high_cardinality_vars = [col for col in df.select_dtypes(include=["object"]).columns
                             if df[col].nunique() > 20]
    
    # Combine the lists of variables to drop and drop them from the dataframe
    cols_to_drop = list(set(high_mi_vars + high_cardinality_vars))
    df = df.drop(columns=cols_to_drop)

    # Fill missing values for numeric columns with the median of the column
    for col in df.select_dtypes(include=["number"]).columns:
        if fill_method == "Median":
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mean())
    
    # Use Label Encoding to convert categorical variables into numeric format
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    return df

# Create a sidebar for Dataset upload or selecting a sample dataset
st.sidebar.header("Load Data")
dataset_choice = st.sidebar.radio(
    "Choose a data source:",
    ["Upload a CSV file", "Use a sample dataset"]
)

# Load data based on the user's source selection
df = None
feature_variables = []

# Allows users to browse their machine for a CSV file and load it into the app (limit 200mb)
if dataset_choice == "Upload a CSV file":
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Dataset loaded successfully!")

# Preload 3 sample datasets for users to explore the models without uploading their own data. 
else:
    sample_choice = st.sidebar.selectbox(
        "Select a sample dataset:",
        ["Breast Cancer", "Titanic"]
    )
    if sample_choice == "Breast Cancer":
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer(as_frame=True)
        df = data.frame
    elif sample_choice == "Titanic":
        df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# Create a sidebar for data preprocessing parameters (missing value threshold and filling method)
st.sidebar.header("Data Preprocessing")
# Create a slider to allow users to set the threshold for dropping columns with n% of missing values (default 40%)
missing_threshold = st.sidebar.slider(
    "Drop columns missing more than:",
    min_value = 0,
    max_value = 100,
    value = 40,
    step = 5,
    format="%d%%"
) / 100  # Convert percentage to a decimal for processing
# Create a radio button to allow users to choose how to fill missing values (median or mean)
fill_method = st.sidebar.radio(
    "Fill missing values with:",
    ["Median", "Mean"]
)

# Display a preview of the loaded dataset and report its shape (number of rows and columns) after preprocessing.
if df is not None:
    df = preprocess_data(df, missing_threshold, fill_method)  # Preprocess the data with the function define on lines 25-46
    st.subheader("Dataset Preview")
    st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    st.dataframe(df.head(10))

# Create a sidebar for feature selection within the dataset
if df is not None:
    st.sidebar.header("Select Feature Variables")
    feature_variables = st.sidebar.multiselect(
        "Select feature variables for clustering and PCA:",
        options=df.columns.tolist(),
        default=df.columns.tolist()  # Default to all columns selected
    )

# Create a sidebar to allow users to select which technique they would like to use and learn about. 
if df is not None and len(feature_variables) >= 2:
    st.sidebar.header("Select an Unsupervised ML Technique to Explore!")
    technique_choice = st.sidebar.selectbox(
        "Choose a technique to apply:",
        ["Principal Component Analysis", "K-Means Clustering", "Hierarchical Clustering"]
    )

# Scale the data and create hyperpareter tuning options for each technique in the main area of the app.
if df is not None and len(feature_variables) >= 2 and technique_choice is not None:

    #Scale the feature variables using StandardScaler to ensure that all features contribute equally to the analysis
    X = df[feature_variables].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Principal Component Analysis (PCA)
    if technique_choice == "Principal Component Analysis":
        st.header("Principal Component Analysis (PCA)")
        st.write("PCA is a dimensionality reduction technique that compresses many features down into a smaller set of components, each capturing a portion of the total variance in the data. It is useful for visualizing high-dimensional data and identifying the most significant variance patterns.")
        
        with st.expander("How does Principal Component Analysis work?"):
            st.write("PCA:\n"
             "1. Standardizes the data so all features are on the same scale.\n"
             "2. Finds the direction of greatest variance in the data: transforming this variance into Principal Component 1.\n"
             "3. Finds the next direction of greatest remaining variance which becomes Principal Component 2, and so on.\n"
             "4. Projects the data onto these new axes to reduce dimensions while keeping as much information as possible.")

        # Add hyperparameter tuning options for PCA in the main area of the app    
        st.subheader("PCA Hyperparameter Tuning")
        # Create a slider to allow users to select the number of principal components to compute for PCA (default 2, range 1-10)
        pca_n_components = st.slider(
            "Select the number of principal components:",
            min_value=1,
            max_value=min(len(feature_variables), 10),
            value=2,
            step=1
        )
        st.caption("The number of principal components to compute. This determines how many dimensions the data will be reduced to. Choosing 2 or 3 components allows for easy visualization, while more components can capture more variance but may be harder to interpret.")

        # Create a button to run the PCA algorithm with the selected hyperparameters
        run_button_pca = st.button("Run PCA", type= "primary")
        if run_button_pca:
            # Fit the PCA model to the scaled data with the selected number of components
            pca = PCA(n_components=pca_n_components)
            X_pca = pca.fit_transform(X_scaled)
            # Calculate the explained variance ratio to understand how much variance is captured by the selected components
            explained_variance = pca.explained_variance_ratio_
            # Calculate the cumulative explained variance to understand how much total variance is captured as you increase the number of components
            cumulative_variance = np.cumsum(explained_variance) * 100  

            st.success("PCA model fitted successfully!")

            # Create a metric summary for selected number of components
            st.subheader("PCA Summary")
            col1, col2 = st.columns(2)
            col1.metric("Number of Components", pca_n_components)
            col2.metric("Cumulative Explained Variance", f"{cumulative_variance[-1]:.1f}%",
                        help="The cumulative variance explained by all of the selected components.")
            
            # Create two subplots to show the explained variance ratio for each component and the cumulative explained variance across n # of components
            plt.figure(figsize=(12, 5))

            #Create the explained variance ratio plot to visualize how much variance is captured by each component
            plt.subplot(1, 2, 1)
            plt.bar(range(1, len(explained_variance) + 1), explained_variance * 100)
            plt.xlabel("Principal Component")
            plt.ylabel("Explained Variance (%)")
            plt.title("Explained % Variance by Each Principal Component")
            plt.xticks(range(1, len(explained_variance) + 1))
            plt.grid(True, axis = 'y')

            # Create the cumulative explained variance plot to visualize how much total variance is captured as you increase the number of components
            plt.subplot(1, 2, 2)
            plt.plot(range(1, len(explained_variance) + 1), np.cumsum(explained_variance) * 100, marker='o')
            plt.xlabel("Number of Principal Components")
            plt.ylabel("Cumulative Explained Variance (%)")
            plt.title("Cumulative Explained Variance (%) by Principal Components")
            plt.xticks(range(1, len(cumulative_variance)+1))
            plt.grid(True)

            plt.tight_layout()
            st.pyplot(plt)

            # Create 2 dropdown explainers to explain how to interpret the explained variance ratio plot and the cumulative explained variance plot
            col1, col2 = st.columns(2)
            with col1:
                with st.expander("How to interpret the Explained Variance Ratio plot?"):
                    st.write("Each bar shows the % of total variance in the data that is captured by that principal component. \n"
                             "Principal Component 1 captures the most variance, followed by Principal Component 2, and so on. \n"
                             "**Look for components where the % of explained variance dropps of significantly. This can help determine how many components to select for dimensionality reduction in order to capture the most variance with the least dimensions.**")
            with col2:
                with st.expander("How to interpret the Cumulative Explained Variance plot?"):
                    st.write("The line plot shows the cumulative % of total variance captured as you increase the number of principal components. \n"
                             "As you add more components, the cumulative explained variance increases, but with diminishing returns. \n"
                             "**Look for the point where adding more components results in only a small increase in cumulative explained variance (the 'elbow' point). This can help determine the optimal number of components to select for dimensionality reduction while retaining most of the variance in the data.**")
                    
            # Create a PCA Loadings heatmap to visualize how the original features contribute to the principal components
            # Build a dataframe with the loading weights for each feature on each principal component
            loadings_df =pd.DataFrame(
                pca.components_,
                columns= feature_variables,
                index=[f"PC{i+1}" for i in range(pca_n_components)]
            )        

            # Limit to top 10 most influential features for each component to keep the plot readable
            top_n = min(10, len(feature_variables))
            # Select the top features based on the absolute loading weights for across all components
            top_features = loadings_df.abs().max(axis=0).nlargest(top_n).index
            loadings_df_top = loadings_df[top_features]

            # Create a heatmap to visualize the loading weights of each feature on each principal component: red indicates a strong positive contribution while blue indicates a strong negative contribution.
            plt.figure(figsize=(10, 6))
            sns.heatmap(loadings_df_top, annot=True, cmap="coolwarm", center=0)
            plt.title("PCA Loadings Heatmap: Feature Contributions to Principal Components\n(10 most influential features shown)")
            plt.xlabel("Original Feature Variables")
            plt.ylabel("Principal Components")
            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()
            st.pyplot(plt)  

            # Create a dropdown explainer to explain how to interpret the PCA loadings heatmap
            with st.expander("How to interpret the PCA Loadings Heatmap?"):
                st.write("The PCA Loadings Heatmap shows how much each original feature contributes to each principal component. \n"
                         "Red cells indicate a strong positive contribution, while blue cells indicate a strong negative contribution. \n"
                         "Features with higher absolute loading values have a greater influence on the corresponding principal component. \n"
                         "**Look for features with high absolute loadings on the first few principal components, as these features are driving the greatest variance captured by those components and can provide insights into the underlying structure of the data.**")


    # K-Means Clustering
    elif technique_choice == "K-Means Clustering":
        st.header("K-Means Clustering")
        st.write("K-Means is a clustering algorithm that partitions the data in k clusters by minimizing the within-cluster sum of squared distances from each point to its assigned cluster centroid. It is used to create groups of similar data points based on their features.")
        
        with st.expander("How does K-means Clustering work?"):
            st.write("The Model:\n"
                     "1. Chooses the number of clusters (k) and randomly place the centroids. These serve as the initial centers for each cluster.\n"
                     "2. Assigns each data point to the nearest centroid (based on a distance metric) to form clusters.\n"
                     "3. Updates the centroids by calculating the mean of the data points in each cluster and placing them at those locations.\n"
                     "4. Repeats steps 2 and 3 until convergence (when centroids do not change significantly) or until a maximum number of iterations is reached.")
        
        # Add hyperparameter tuning options for K-Means clustering in the main area of the app
        st.subheader("K-Means Hyperparameter Tuning")
        # Create a slider to allow users to select the number of clusters (k) for K-Means clustering (default 3, range 2-15)
        km_k = st.slider(
            "Select the number of clusters (k):",
            min_value=2,
            max_value=15,
            value=3,
            step=1
        )
        st.caption("The number of clusters the model will create. It is important to experiment with different values of k to find the optimal number of clusters for your data. Pay attention to the elbow plot and silhoette score after running the model to help determine the best k value.")

        # Create a slider to allow users to select the maximum number of iterations for K-Means clustering (default 300, range 100-500)
        km_max_iter = st.slider(
            "Select the maximum number of iterations:",
            min_value=100,
            max_value=500,
            value=300,
            step=50
        )
        st.caption("The maximum number of iterations the algorithm will run for. If the algorithm does not converge before reaching this number, it will stop and return the current cluster assignments. Increasing this value can help ensure convergence, but may also increase computation time.")

        # Create a button to run the K-Means clustering algorithm with the selected hyperparameters
        run_button_km = st.button("Run K-Means", type= "primary")

        if run_button_km:
            kmeans = KMeans(n_clusters=km_k, max_iter=km_max_iter, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)

            st.success("K-Means model fitted successfully!")

            # Create a metric summary for selected k
            st.subheader("Cluster Summary")
            col1, col2, col3 = st.columns(3)
            col1.metric("Number of Clusters", km_k)
            col2.metric("Inertia (WCSS)", f"{kmeans.inertia_:.1f}",
                        help="Total within-cluster sum of squares. Lower is better: look to elbow plot for more details.")
            col3.metric("Silhouette Score", f"{silhouette_score(X_scaled, clusters):.3f}",
                        help="Ranges from -1 to 1. Higher values mean clusters are dense and well-separated: look to silhouette score plot for more details.")
            

            # PCA Cluster Visualization (use PCA to reduce the dimensionality of the data to 2 components for visualization purposes)
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            # Create a scatter plot of the PCA components colored by the cluster assignments
            plt.figure(figsize=(10, 6))
            # Iterate over unique cluster labels and plot each cluster with a different color
            for cluster_label in np.unique(clusters):
                indices = np.where(clusters == cluster_label)
                plt.scatter(X_pca[indices, 0], X_pca[indices, 1], label=f"Cluster {cluster_label}")
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            plt.title("K-Means Clustering Visualization (using PCA to reduce dimensions)")
            plt.legend(loc="best")
            plt.grid(True)
            st.pyplot(plt)
            # Add an explainer dropdown to explain how to interpret the PCA cluster visualization
            with st.expander("How to interpret the PCA cluster visualization?"):
                st.write("The PCA cluster visualization is a scatter plot of the data points in the space of the first two principal components. Each point represents a data point from the original dataset, and the color of the point indicates which cluster it belongs to based on the K-Means algorithm. The distance between points in this plot reflects their similarity in terms of the original features, with points that are closer together being more similar. Look for distinct groups of points that are colored the same; The more distinct the clusters appear in this plot, the better the clustering performance.")

            st.subheader("Determine the Optimal Number of Clusters with WCSS and Silhouette Scores")
            # Calculate the Within-Cluster Sum of Squares (WCSS) and silhouette score for all available k values from 2 to 15
            # Define range of possible k values to evaluate
            ks = range(2, 16)
            wcss = []
            silhouette_scores = []

            # Loop through each k value, fit the KMeans model, and calculate WCSS and silhouette score
            for k in ks:
                kmeans = KMeans(n_clusters=k, max_iter=km_max_iter, random_state=42)
                kmeans.fit(X_scaled)
                wcss.append(kmeans.inertia_)  # Inertia is the WCSS for the fitted model
                silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_)) # Calculate silhouette score for the fitted model

            # Create a figure with two subplots: one for the elbow plot and one for the silhouette scores
            # Create the elbow plot to visualize WCSS difference across k values
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            plt.plot(ks, wcss, marker='o')
            plt.xlabel("Number of Clusters (k)")
            plt.ylabel("Within-Cluster Sum of Squares (WCSS)")
            plt.title("Elbow Plot for K-Means Clustering")
            plt.grid(True)

            # Create the silhouette score plot to visualize silhouette scores across k values
            plt.subplot(1, 2, 2)
            plt.plot(ks, silhouette_scores, marker='o')
            plt.xlabel("Number of Clusters (k)")
            plt.ylabel("Silhouette Score")
            plt.title("Silhouette Scores for K-Means Clustering")
            plt.grid(True)

            plt.tight_layout()
            st.pyplot(plt)

            # Add explainer dropdowns to explain how to interpret the elbow plot and silhouette score plot
            col1, col2 = st.columns(2)
            with col1:
                with st.expander("How to interpret the Elbow Plot?"):
                    st.write("The Elbow Plot displays the Within-Cluster Sum of Squares (WCSS) values between different values of k.\n"
                             "The WCSS measures the total distance between each point and its assigned cluster centroid, with lower values indicating tighter clusters. As you increase k, WCSS will decrease because more clusters will better fit the data.\n "
                             "**The 'elbow' point on the plot, where the rate of decrease sharply changes, is often considered the optimal k value, as it indicates a balance between cluster compactness and model simplicity.**")
            with col2:
                with st.expander("How to interpret the Silhouette Score Plot?"):
                    st.write("The Silhouette Score Plot displays the silhouette score for the different values of k.\n"
                             "Silhouette score measures how similar a data point is to its own cluster compared to other clusters. A higher silhouette score indicates that the data points are well matched to their own cluster and poorly matched to neighboring clusters.\n"
                             "**When interpreting the plot, keep in mind the elbow plot's best k value, and look for a k value that is close to that value and has a high silhouette score, as this indicates that the clusters are well-defined and distinct from each other.**")
    # Heirarchical Clustering
    elif technique_choice == "Hierarchical Clustering":
        st.header("Hierarchical Clustering")
        st.write("Hierarchical Clustering builds a tree of clusters by successively merging the most similar points/clusters together. The result is a dendrogram (a tree-like structure) that shows the full history of all the merges. By cutting the tree at a chosen level, you can select the number of clusters that best fits your data.")

        with st.expander("How does Hierarchical Clustering work?"):
            st.write("The Model:\n"
                     "1. Starts with each data point as its own cluster (n clusters for n data points).\n"
                     "2. Calculates the distance between all pairs of clusters and merges the two closest clusters together.\n"
                     "3. Repeats step 2 until all points are merged into a single cluster, creating a hierarchy of clusters.\n"
                     "4. The resulting dendrogram can be cut at different levels to select the desired number of clusters based on the distance between merges.")
            
        # Add hyperparameter tuning options for hierarchical clustering in the main area of the app
        st.subheader("Hierarchical Clustering Hyperparameter Tuning")
        # Create a slider to allow users to select the k value to cut the dendrogram for hierarchical clustering (default 3, range 2-15)
        hc_k = st.slider(
            "Select the number of clusters (k) to cut the dendrogram:",
            min_value=2,
            max_value=15,
            value=3,
            step=1
        )
        st.caption("The number of clusters to form by cutting the dendrogram. It is important to experiment with different values of k to find the optimal number of clusters for your data. Pay attention to the dendrogram and silhouette scores after running the model to help determine the best k value.")

        # Create a slider to allow users to select the linkage method for hierarchical clustering (default 'ward', options 'ward', 'complete', 'average', 'single')
        hc_linkage = st.selectbox(
            "Select the linkage method for hierarchical clustering:",
            ["ward", "complete", "average", "single"],
            index=0
        )
        st.caption("The linkage method determines how the distance between clusters is calculated when merging them together. 'Ward' minimizes the variance within clusters, 'complete' uses the maximum distance between points in clusters, 'average' uses the average distance between points in clusters, and 'single' uses the minimum distance between points in clusters. The choice of linkage method can affect the shape and size of the resulting clusters, so it is important to experiment with different options to find the best fit for your data. 'Ward' is often a good default choice for numerical data, while 'complete' or 'average' can be better for categorical data or when you want more compact clusters.")

        # Create a button to run the hierarchical clustering algorithm with the selected hyperparameters
        run_button_hc = st.button("Run Hierarchical Clustering", type= "primary")
        if run_button_hc:
            # Limit the dendrogram to the top 100 most similar merges to keep the plot readable
            n_dendro = min(100, len(X_scaled))
            # Build linkage matrix for the dendrogram vizualization using the selected linkage method
            Z = linkage(X_scaled[:n_dendro], method=hc_linkage)

            # Fit the complete model and get cluster lables based on the hyperparameters selected by the user
            agg = AgglomerativeClustering(n_clusters=hc_k, linkage=hc_linkage)
            hc_labels = agg.fit_predict(X_scaled)
            # Calculate the silhouette score for the hierarchical clustering model to evaluate cluster quality
            hc_silhouette = silhouette_score(X_scaled, hc_labels)

            st.success("Hierarchical Clustering model fitted successfully!")

            # Create a metric summary for hierarchical clustering results
            st.subheader("Cluster Summary")
            col1, col2 = st.columns(2)
            col1.metric("Number of Clusters", hc_k)
            col2.metric("Silhouette Score", f"{hc_silhouette:.3f}",
                        help="Ranges from -1 to 1. Higher values mean clusters are dense and well-separated: look to silhouette score plot for more details.")
            
            # PCA Cluster Visualization (use PCA to reduce the dimensionality of the data to 2 components for visualization purposes)
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            # Create a scatter plot of the PCA components colored by the cluster assignments
            plt.figure(figsize=(10, 6))
            # Iterate over unique cluster labels and plot each cluster with a different color
            for cluster_label in np.unique(hc_labels):
                indices = np.where(hc_labels == cluster_label)
                plt.scatter(X_pca[indices, 0], X_pca[indices, 1], label=f"Cluster {cluster_label}")
            plt.xlabel("Principal Component 1")
            plt.ylabel("Principal Component 2")
            plt.title("Hierarchical Clustering Visualization (using PCA to reduce dimensions)")
            plt.legend(loc="best")
            plt.grid(True)
            st.pyplot(plt)
            # Add an explainer dropdown to explain how to interpret the PCA cluster visualization
            with st.expander("How to interpret the PCA cluster visualization?"):
                st.write("The PCA cluster visualization is a scatter plot of the data points in the space of the first two principal components. Each point represents a data point from the original dataset, and the color of the point indicates which cluster it belongs to based on the hierarchical clustering algorithm. The distance between points in this plot reflects their similarity in terms of the original features, with points that are closer together being more similar. Look for distinct groups of points that are colored the same; The more distinct the clusters appear in this plot, the better the clustering performance.")

            st.subheader("Determine the Optimal Number of Clusters with the Dendrogram and Silhouette Scores")
            # Create a dendrogram to visualize the hierarchical clustering process and the distance between merges
            plt.figure(figsize=(14, 6))
            dendrogram(Z)
            plt.title(f"Hierarchical Clustering Dendrogram ({hc_linkage} linkage, showing top {n_dendro} merges)")
            plt.xlabel("Sample Index")
            plt.ylabel("Distance")
            st.pyplot(plt)
            with st.expander("How to read the Dendrogram?"):
                st.write("Each leaf at the bottom represents a single data point.\n"
                         "As you move up the tree, points and clusters are merged together.\n"
                         "The height of each merge on the y-axis shows how different the two clusters were meaning taller merges show more dissimilar clusters.\n"
                         "**Look for the largest vertical gaps in the dendrogram. The number of vertical lines a horizontal cut through that gap crosses is your optimal k.**")


            # Add a silhouette score plot to evaluate the quality of the hierarchical clustering results across different k values
            ks = range(2, 11)
            sil_scores = []
            # Loop through each k value, fit the AgglomerativeClustering model, and calculate silhouette score
            for k in ks:
                agg = AgglomerativeClustering(n_clusters=k, linkage=hc_linkage)
                labels = agg.fit_predict(X_scaled)
                sil_scores.append(silhouette_score(X_scaled, labels))
            # Plot the appended silhouette scores
            plt.figure(figsize=(8, 4))
            plt.plot(ks, sil_scores, marker='o')
            plt.xlabel("Number of Clusters (k)")
            plt.ylabel("Silhouette Score")
            plt.title("Silhouette Scores for Hierarchical Clustering")
            plt.grid(True)    
            st.pyplot(plt)
            with st.expander("How to interpret the Silhouette Score Plot?"):
                st.write("The Silhouette Score Plot displays the silhouette score for the different values of k.\n"
                         "Silhouette score measures how similar a data point is to its own cluster compared to other clusters. A higher silhouette score indicates that the data points are well matched to their own cluster and poorly matched to neighboring clusters.\n"
                         "**Look for the k with the highest score and cross-reference with the dendrogram to confirm it aligns with a natural gap in the tree.**")