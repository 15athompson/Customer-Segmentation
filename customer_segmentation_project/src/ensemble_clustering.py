import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from src.advanced_clustering import hierarchical_clustering, gaussian_mixture_model

def ensemble_clustering(data, n_clusters=5):
    """
    Perform ensemble clustering using multiple techniques.
    """
    # Perform clustering using different methods
    kmeans_labels = KMeans(n_clusters=n_clusters, random_state=42).fit_predict(data)
    hierarchical_labels = hierarchical_clustering(data, n_clusters)
    gmm_labels = gaussian_mixture_model(data, n_components=n_clusters)
    
    # Combine labels into a single feature matrix
    ensemble_features = np.column_stack((kmeans_labels, hierarchical_labels, gmm_labels))
    
    # Train a Random Forest classifier on the ensemble features
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(ensemble_features, kmeans_labels)  # Use KMeans labels as a baseline
    
    # Generate final cluster labels
    final_labels = rf_classifier.predict(ensemble_features)
    
    return final_labels
