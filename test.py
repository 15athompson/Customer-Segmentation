import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging
import json
from flask import Flask, render_template, request, jsonify
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA

def evaluate_clusters(X, labels):
    """
    Evaluates the clustering performance using silhouette score.
    :param X: Preprocessed data.
    :param labels: Cluster labels.
    :return: Silhouette score.
    """
    score = silhouette_score(X, labels)
    return score
from sklearn import __version__ as sklearn_version
from packaging import version

def load_config(config_file):
    """
    Load configuration from a JSON file.

    Args:
        config_file (str): Path to the configuration file.

    Returns:
        dict: Configuration parameters.
    """
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Config file {config_file} not found.")
        raise
    except json.JSONDecodeError:
        logging.error(f"Error decoding {config_file}. Please ensure it's valid JSON.")
        raise

def load_data(file_name):
    """
    Load customer data from a CSV file.

    Args:
        file_name (str): Name of the CSV file in the data directory.

    Returns:
        pd.DataFrame: Loaded customer data.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the required columns are missing from the CSV file.
    """
    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate to the parent directory (project root)
    project_root = os.path.dirname(current_dir)
    # Construct the full path to the data file
    file_path = os.path.join(project_root, file_name)
    
    print(f"Current working directory: {os.getcwd()}")
    print(f"Attempting to load file: {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    df = pd.read_csv(file_path)

    required_columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"The CSV file is missing one or more required columns: {required_columns}")

    return df

def preprocess_data(df, remove_outliers=False, categorical_columns=None):
    """
    Preprocess the data by handling missing values, outliers, scaling features, and encoding categorical variables.

    Args:
        df (pd.DataFrame): Input DataFrame containing customer data.
        remove_outliers (bool): Whether to remove outliers using IQR method.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Preprocessed feature matrix.
            - list: List of feature names after preprocessing.
    """
    # Handle missing values (if any)
    df = df.dropna()
    
    # Select features for segmentation
    numerical_columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    
    X = df[numerical_columns]
    
    if remove_outliers:
        for feature in numerical_columns:
            Q1 = X[feature].quantile(0.25)
            Q3 = X[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            X = X[(X[feature] >= lower_bound) & (X[feature] <= upper_bound)]
    
    # Create preprocessing steps
    numeric_transformer = StandardScaler()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_columns)
        ])
    
    # Fit and transform the data
    X_preprocessed = preprocessor.fit_transform(X)
    
    # Get feature names after preprocessing
    feature_names = numerical_columns
    
    return X_preprocessed, feature_names

def perform_kmeans(X, n_clusters):
    """
    Perform K-means clustering on the data.

    Args:
        X (np.ndarray): Scaled feature matrix.
        n_clusters (int): Number of clusters to form.

    Returns:
        KMeans: Fitted K-means model.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X)
    return kmeans

def perform_dbscan(X, eps=0.5, min_samples=5):
    """
    Perform DBSCAN clustering on the data.

    Args:
        X (np.ndarray): Scaled feature matrix.
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
        DBSCAN: Fitted DBSCAN model.
    """
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    dbscan.fit(X)
    return dbscan

def perform_gmm(X, n_components):
    """
    Perform Gaussian Mixture Model clustering on the data.

    Args:
        X (np.ndarray): Scaled feature matrix.
        n_components (int): Number of mixture components.

    Returns:
        GaussianMixture: Fitted GMM model.
    """
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X)
    return gmm

def perform_hierarchical(X, n_clusters):
    """
    Perform Hierarchical Clustering on the data.

    Args:
        X (np.ndarray): Scaled feature matrix.
        n_clusters (int): Number of clusters.

    Returns:
        AgglomerativeClustering: Fitted Hierarchical Clustering model.
    """
    hierarchical = AgglomerativeClustering(n_clusters=n_clusters)
    hierarchical.fit(X)
    return hierarchical

def analyze_feature_importance(X, y, features):
    """
    Analyze feature importance using Random Forest and permutation importance.

    Args:
        X (np.ndarray): Scaled feature matrix.
        y (np.ndarray): Cluster labels.
        features (list): List of feature names.

    Returns:
        pd.DataFrame: DataFrame with feature importances.
    """
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    perm_importance = permutation_importance(rf, X, y, n_repeats=10, random_state=42)

    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': rf.feature_importances_,
        'permutation_importance': perm_importance.importances_mean
    })
    feature_importance = feature_importance.sort_values('permutation_importance', ascending=False)
    return feature_importance

def find_optimal_clusters(X, max_clusters=10):
    """
    Find the optimal number of clusters using the elbow method.

    Args:
        X (np.ndarray): Scaled feature matrix.
        max_clusters (int): Maximum number of clusters to consider.

    Returns:
        int: Optimal number of clusters.
    """
    inertias = []
    silhouette_scores = []

    for k in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))

    # Plot the elbow curve
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(2, max_clusters + 1), inertias, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow Method')

    plt.subplot(1, 2, 2)
    plt.plot(range(2, max_clusters + 1), silhouette_scores, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Analysis')

    plt.tight_layout()
    plt.show()

    # Choose the optimal number of clusters based on the elbow point
    optimal_clusters = np.argmin(np.diff(inertias)) + 2
    return optimal_clusters

def visualize_clusters(X, kmeans, dbscan, gmm, hierarchical, features):
    """
    Visualize the clusters in interactive 3D scatter plots for all clustering methods.
    """
    methods = [('K-means', kmeans.labels_),
               ('DBSCAN', dbscan.labels_),
               ('GMM', gmm.predict(X)),
               ('Hierarchical', hierarchical.labels_)]
    
    figs = []
    for method, labels in methods:
        fig = px.scatter_3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], color=labels,
                            labels={'x': features[0], 'y': features[1], 'z': features[2]},
                            title=f'{method} Clustering')
        figs.append(fig)
    
    return figs

def save_results(df, output_file):
    """
    Save the clustered data to a CSV file.

    Args:
        df (pd.DataFrame): DataFrame with cluster labels.
        output_file (str): Path to the output CSV file.
    """
    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

def profile_clusters(df):
    """
    Generate a summary of each cluster, including PCA components if available.

    Args:
        df (pd.DataFrame): DataFrame with cluster labels and possibly PCA components.

    Returns:
        pd.DataFrame: Summary statistics for each cluster.
    """
    summary_columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    pca_columns = [col for col in df.columns if col.startswith('PC')]
    summary_columns.extend(pca_columns)

    cluster_summary = df.groupby('Cluster').agg({
        col: ['mean', 'min', 'max'] for col in summary_columns
    })
    
    cluster_summary.columns = ['_'.join(col).strip() for col in cluster_summary.columns.values]
    return cluster_summary

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/segment', methods=['POST'])
def segment():
    # Get data from the form
    data = request.json
    # Process the data (you'll need to implement this)
    result = process_data(data)
    return jsonify(result)

def process_data(data):
    # Implement data processing and segmentation here
    # This is a placeholder function
    return {"result": "Data processed successfully"}

def create_interactive_plot(X, labels, features):
    fig = px.scatter_3d(x=X[:, 0], y=X[:, 1], z=X[:, 2], color=labels,
                        labels={'x': features[0], 'y': features[1], 'z': features[2]})
    return fig.to_json()

def perform_pca(X, n_components=3):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return X_pca, pca.explained_variance_ratio_

def main():
    parser = argparse.ArgumentParser(description="Customer Segmentation Analysis")
    parser.add_argument("--config", default="config.json", help="Path to configuration file")
    parser.add_argument("--remove-outliers", action="store_true", help="Remove outliers during preprocessing")
    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    df = None  # Initialize df outside the try block

    try:
        # Get the directory of the current script
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Navigate to the parent directory (project root)
        project_root = os.path.dirname(current_dir)
        # Construct the full path to the config file
        config_path = os.path.join(project_root, args.config)
        
        # Load configuration
        config = load_config(config_path)
        
        # Load data
        df = load_data(config['data_file'])
        
        if df is not None:
            # Preprocess data
            X_preprocessed, features = preprocess_data(df, remove_outliers=args.remove_outliers)
            
            # Perform PCA
            X_pca, explained_variance_ratio = perform_pca(X_preprocessed)
            logging.info(f"Explained variance ratio: {explained_variance_ratio}")
            
            # Find optimal number of clusters
            optimal_clusters = find_optimal_clusters(X_pca, max_clusters=config['max_clusters'])
            logging.info(f"Optimal number of clusters: {optimal_clusters}")
            
            # Perform clustering on PCA-transformed data
            kmeans = perform_kmeans(X_pca, optimal_clusters)
            dbscan = perform_dbscan(X_pca, eps=config['dbscan_eps'], min_samples=config['dbscan_min_samples'])
            gmm = perform_gmm(X_pca, optimal_clusters)
            hierarchical = perform_hierarchical(X_pca, optimal_clusters)
            
            # Evaluate clusters
            kmeans_silhouette = evaluate_clusters(X_pca, kmeans.labels_)
            dbscan_silhouette = evaluate_clusters(X_pca, dbscan.labels_)
            gmm_silhouette = evaluate_clusters(X_pca, gmm.predict(X_pca))
            hierarchical_silhouette = evaluate_clusters(X_pca, hierarchical.labels_)
            logging.info(f"K-means Silhouette Score: {kmeans_silhouette:.4f}")
            logging.info(f"DBSCAN Silhouette Score: {dbscan_silhouette:.4f}")
            logging.info(f"GMM Silhouette Score: {gmm_silhouette:.4f}")
            logging.info(f"Hierarchical Clustering Silhouette Score: {hierarchical_silhouette:.4f}")
            
            # Visualize the clusters
            figs = visualize_clusters(X_pca, kmeans, dbscan, gmm, hierarchical, ['PC1', 'PC2', 'PC3'])
            
            # Save interactive plots
            for i, fig in enumerate(figs):
                fig.write_html(f"cluster_plot_{i}.html")
            
            # Add cluster labels to the original dataframe
            df['KMeans_Cluster'] = kmeans.labels_
            df['DBSCAN_Cluster'] = dbscan.labels_
            df['GMM_Cluster'] = gmm.predict(X_pca)
            df['Hierarchical_Cluster'] = hierarchical.labels_
            
            # Analyze feature importance for all methods
            kmeans_importance = analyze_feature_importance(X_pca, kmeans.labels_, ['PC1', 'PC2', 'PC3'])
            dbscan_importance = analyze_feature_importance(X_pca, dbscan.labels_, ['PC1', 'PC2', 'PC3'])
            gmm_importance = analyze_feature_importance(X_pca, gmm.predict(X_pca), ['PC1', 'PC2', 'PC3'])
            hierarchical_importance = analyze_feature_importance(X_pca, hierarchical.labels_, ['PC1', 'PC2', 'PC3'])
            
            logging.info("\nK-means Feature Importance:")
            logging.info(kmeans_importance)
            logging.info("\nDBSCAN Feature Importance:")
            logging.info(dbscan_importance)
            logging.info("\nGMM Feature Importance:")
            logging.info(gmm_importance)
            logging.info("\nHierarchical Clustering Feature Importance:")
            logging.info(hierarchical_importance)
            
            # Profile clusters for all methods
            kmeans_summary = profile_clusters(df.rename(columns={'KMeans_Cluster': 'Cluster'}))
            dbscan_summary = profile_clusters(df.rename(columns={'DBSCAN_Cluster': 'Cluster'}))
            gmm_summary = profile_clusters(df.rename(columns={'GMM_Cluster': 'Cluster'}))
            hierarchical_summary = profile_clusters(df.rename(columns={'Hierarchical_Cluster': 'Cluster'}))
            logging.info("K-means Cluster Summary:")
            logging.info(kmeans_summary)
            logging.info("\nDBSCAN Cluster Summary:")
            logging.info(dbscan_summary)
            logging.info("\nGMM Cluster Summary:")
            logging.info(gmm_summary)
            logging.info("\nHierarchical Clustering Summary:")
            logging.info(hierarchical_summary)
            
            # Save results
            save_results(df, config['output_file'])
        else:
            logging.error("Failed to load data. Please check the data file and try again.")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")

    # Add Flask app run at the end of main()
    app.run(debug=True)

if __name__ == "__main__":
    main()