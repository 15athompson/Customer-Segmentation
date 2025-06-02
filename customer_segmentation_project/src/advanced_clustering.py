import numpy as np
from sklearn.cluster import AgglomerativeClustering, GaussianMixture
from sklearn.mixture import BayesianGaussianMixture

def hierarchical_clustering(data, n_clusters=5):
    """
    Perform hierarchical clustering on the input data.
    
    :param data: numpy array of shape (n_samples, n_features)
    :param n_clusters: number of clusters to form
    :return: cluster labels
    """
    clustering = AgglomerativeClustering(n_clusters=n_clusters)
    return clustering.fit_predict(data)

def gaussian_mixture_model(data, n_components=5):
    """
    Perform Gaussian Mixture Model clustering on the input data.
    
    :param data: numpy array of shape (n_samples, n_features)
    :param n_components: number of mixture components
    :return: cluster labels
    """
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    return gmm.fit_predict(data)

def bayesian_gaussian_mixture_model(data, n_components=5):
    """
    Perform Bayesian Gaussian Mixture Model clustering on the input data.
    
    :param data: numpy array of shape (n_samples, n_features)
    :param n_components: maximum number of mixture components
    :return: cluster labels
    """
    bgmm = BayesianGaussianMixture(n_components=n_components, random_state=42)
    return bgmm.fit_predict(data)

# Add more advanced clustering techniques as needed
