# Import necessary libraries and tools for data manipulation, machine learning, and visualization
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Setup Streamlit app title and description
st.set_page_config(page_title="Branches or Neighborhoods: An Interactive Supervised ML App", layout="wide")

st.title("Branches or Neighborhoods: An Interactive Supervised ML App")
st.write("Upload your own dataset or explore one of three sample datasets, tune the hyperparameters, and compare how a Decision Tree and K-Nearest Neighbors classifier perform side-by-side. Which model fits your data better: branches or neighborhoods?")

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
st.sidebar.header("1. Load Data")
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
        ["BRFSS Diabetes Indicators", "Breast Cancer", "Titanic"]
    )
    if sample_choice == "BRFSS Diabetes Indicators":
        import os
        df = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"))
        df = df.sample(n=5000, random_state=42) # Sample for faster processing
    elif sample_choice == "Breast Cancer":
        from sklearn.datasets import load_breast_cancer
        data = load_breast_cancer(as_frame=True)
        df = data.frame
    elif sample_choice == "Titanic":
        df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# Create a sidebar for data preprocessing parameters (missing value threshold and filling method)
st.sidebar.header("2. Data Preprocessing")
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

# Prompt users to select the target and feature variables
if df is not None:
    st.sidebar.header("3. Select Target Variable")
    target_variable = st.sidebar.selectbox(
        "Select the target variable for classification:",
        df.columns
    )

    # Prompt users to select features for the model (all columns except the target variable)
    st.sidebar.header("4. Select Features")
    feature_variables = st.sidebar.multiselect(
        "Select feature variables (at least 2):",
        [col for col in df.columns if col != target_variable]
    )

# Display a preview of the loaded dataset and report its shape (number of rows and columns) after preprocessing.
if df is not None:
    df = preprocess_data(df, missing_threshold, fill_method)  # Preprocess the data with the function define on lines 25-46
    st.subheader("Dataset Preview")
    st.write(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    st.dataframe(df.head(10))

# Display hyperparemeter model controls below the dataset preview. These controls will allow users to adjust the parameters of the Decision Tree and KNN models.
if df is not None and len(feature_variables) >= 2:
    st.header("Model Hyperparameters")
    st.write("Adjust the hyperparameters below before training the models.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Decision Tree 🌳")
        dt_max_depth = st.slider("Max Depth", 1, 20, 5)
        st.caption("The maximum depth of the tree. Limiting the depth of the tree can help prevent overfitting, but setting it too low may lead to underfitting.")
        dt_min_samples_split = st.slider("Min Samples Split", 2, 20, 2)
        st.caption("The minimum number of samples required to split an internal node. Increasing this value can lead to a simpler model that may generalize better but may also underfit the training data.")
        dt_criterion = st.selectbox("Criterion", ["gini", "entropy"])
        st.caption("The function to measure the quality of a split. 'gini' measures disorder in each split; Lower is better. Entropy measures diversity using probility; Lower means more homogeneity. Both are used to find the best split at each node in the tree, but they may lead to different splits and different tree structures.")
    with col2:
        st.markdown("#### K-Nearest Neighbors 🏘️")
        knn_n_neighbors = st.slider("Number of Neighbors (K)", 1, 20, 5)
        st.caption("The number of nearest neighbors to use for classification. A smaller K can lead to a more flexible model that may capture more complex patterns but may also be sensitive to noise in the data, while a larger K can lead to a smoother decision boundary that may generalize better but may also miss important patterns.")
        knn_weights = st.selectbox("Weights", ["uniform", "distance"])
        st.caption("The weight function used in prediction. 'uniform' means that all neighbors are weighted equally, while 'distance' means that closer neighbors will have a greater influence on the prediction than farther neighbors.")

# Add button to train models
if df is not None:
    st.sidebar.header("5. Train Models")
    train_button = st.sidebar.button("Train Models", use_container_width=True, type="primary")


# Train models when user clicks the Train Models button
if df is not None:
    if train_button:
        if len(feature_variables) < 2:
            st.error("Please select at least 2 feature variables.")
        else:
            X = df[feature_variables]
            y = df[target_variable]

            # Split the data into training and testing sets (80% train, 20% test)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Standardize the features for KNN (KNN is distance based so it is neccessary to scale the features in order to insure that the model is not innaccurately weighted based on feature scales)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train Decision Tree model with user-selected hyperparameters
            dt_model = DecisionTreeClassifier(max_depth=dt_max_depth, min_samples_split=dt_min_samples_split, criterion=dt_criterion, random_state=42)
            dt_model.fit(X_train, y_train)

            # Train K-Nearest Neighbors model with user-selected hyperparameters
            knn_model = KNeighborsClassifier(n_neighbors=knn_n_neighbors, weights=knn_weights)
            knn_model.fit(X_train_scaled, y_train)

            st.success("Models trained successfully!")

            # Generate predictions and evaluate model performance when the models are trained
            # Generate predictions for both models
            dt_predictions = dt_model.predict(X_test)
            knn_predictions = knn_model.predict(X_test_scaled)

            # Calculate evaluation metrics for both models
            # Decision Tree Metrics
            dt_accuracy = accuracy_score(y_test, dt_predictions)
            dt_precision = precision_score(y_test, dt_predictions, average="weighted", zero_division=0)
            dt_recall = recall_score(y_test, dt_predictions, average="weighted", zero_division=0)
            dt_f1 = f1_score(y_test, dt_predictions, average="weighted", zero_division=0)

            # KNN Metrics
            knn_accuracy = accuracy_score(y_test, knn_predictions)
            knn_precision = precision_score(y_test, knn_predictions, average="weighted", zero_division=0)
            knn_recall = recall_score(y_test, knn_predictions, average="weighted", zero_division=0)
            knn_f1 = f1_score(y_test, knn_predictions, average="weighted", zero_division=0)

            # Display evaluation metrics in both columns for comparison
            st.subheader("Model Performance")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Decision Tree 🌳")
                st.metric("Accuracy", f"{dt_accuracy:.2f}", help="The proportion of correct predictions out of all predictions made. Higher is better.")
                st.metric("Precision", f"{dt_precision:.2f}", help="The proportion of true positive predictions out of all positive predictions. It indicates how many of the predicted positive instances are actually positive.")
                st.metric("Recall", f"{dt_recall:.2f}", help="The proportion of true positive predictions out of all actual positive instances. It indicates how many of the actual positive instances were correctly identified.")
                st.metric("F1 Score", f"{dt_f1:.2f}", help="The harmonic mean of precision and recall. It provides a single metric that balances both precision and recall.")

            with col2:
                st.markdown("#### K-Nearest Neighbors 🏘️")
                st.metric("Accuracy", f"{knn_accuracy:.2f}", help="The proportion of correct predictions out of all predictions made. Higher is better.")
                st.metric("Precision", f"{knn_precision:.2f}", help="The proportion of true positive predictions out of all positive predictions. It indicates how many of the predicted positive instances are actually positive.")
                st.metric("Recall", f"{knn_recall:.2f}", help="The proportion of true positive predictions out of all actual positive instances. It indicates how many of the actual positive instances were correctly identified.")
                st.metric("F1 Score", f"{knn_f1:.2f}", help="The harmonic mean of precision and recall. It provides a single metric that balances both precision and recall.")

            # Display confusion matrices for both models
            st.subheader("Confusion Matrices")
            with st.expander("What is a Confusion Matrix?"):
                st.write("A confusion matrix is a table that is used to evaluate the performance of a classification model. It shows the counts of true positive, true negative, false positive, and false negative predictions made by the model. The confusion matrix helps to understand how well the model is performing in terms of correctly classifying instances and where it may be making errors. The true positive (top-left) count represents the number of instances that were correctly predicted as positive, while the true negative (bottom-right) count represents the number of instances that were correctly predicted as negative. The false positive (top-right) count represents the number of instances that were incorrectly predicted as positive, while the false negative (bottom-left) count represents the number of instances that were incorrectly predicted as negative. By analyzing the confusion matrix, you can gain insights into the types of errors your model is making and how to improve its performance.")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Decision Tree 🌳")
                dt_cm = confusion_matrix(y_test, dt_predictions)
                fig, ax = plt.subplots()
                sns.heatmap(dt_cm, annot=True, fmt="d", cmap="Blues", ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)
            
            with col2:
                st.markdown("#### K-Nearest Neighbors 🏘️")
                knn_cm = confusion_matrix(y_test, knn_predictions)
                fig, ax = plt.subplots()
                sns.heatmap(knn_cm, annot=True, fmt="d", cmap="Greens", ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                st.pyplot(fig)

            # Display ROC curves comparing both models
            from sklearn.metrics import roc_curve, roc_auc_score

            st.subheader("ROC Curves")
            with st.expander("What is a ROC Curve?"):
                st.write("A ROC (Receiver Operating Characteristic) curve is a plot of the true positive rate against the false positive rate for different threshold settings. It is used to evaluate the performance of a binary classifier by showing the tradeoffs between higher sensitivity and lower specificity. Better models will have curves that are closer to the top-left corner of the plot, indicating higher true positive rates and lower false positive rates. The area under the ROC curve (AUC) is a single metric that summarizes the overall performance of the model, with higher values indicating better performance. A score of 1 is a perfect model, while a score of 0.5 is the same level as random guessing.")

            # Calculate ROC curve and AUC for Decision Tree
            dt_probs = dt_model.predict_proba(X_test)[:, 1]
            dt_fpr, dt_tpr, _ = roc_curve(y_test, dt_probs)
            dt_auc = roc_auc_score(y_test, dt_probs)

            # Calculate ROC curve and AUC for KNN
            knn_probs = knn_model.predict_proba(X_test_scaled)[:, 1]
            knn_fpr, knn_tpr, _ = roc_curve(y_test, knn_probs)
            knn_auc = roc_auc_score(y_test, knn_probs)

            # Plot ROC curves for both models
            fig, ax = plt.subplots()
            ax.plot(dt_fpr, dt_tpr, label=f"Decision Tree (AUC = {dt_auc:.2f})", color="blue")
            ax.plot(knn_fpr, knn_tpr, label=f"KNN (AUC = {knn_auc:.2f})", color="green")
            ax.plot([0, 1], [0, 1], "k--", label="Random Guess")
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve Comparison")
            ax.legend(loc="lower right")
            st.pyplot(fig)