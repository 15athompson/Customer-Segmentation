import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA

def select_k_best_features(X, y, k=10, score_func=f_classif):
    """
    Select the k best features based on the scoring function.
    
    :param X: feature matrix
    :param y: target variable
    :param k: number of top features to select
    :param score_func: scoring function (f_classif or mutual_info_classif)
    :return: selected feature indices
    """
    selector = SelectKBest(score_func=score_func, k=k)
    selector.fit(X, y)
    return selector.get_support(indices=True)

def mutual_information_feature_selection(X, y, k=10):
    """
    Select the k best features based on mutual information.
    
    :param X: feature matrix
    :param y: target variable
    :param k: number of top features to select
    :return: selected feature indices
    """
    return select_k_best_features(X, y, k, mutual_info_classif)

def pca_feature_selection(X, n_components=0.95):
    """
    Perform PCA for feature selection/dimensionality reduction.
    
    :param X: feature matrix
    :param n_components: number of components to keep or variance threshold
    :return: transformed feature matrix
    """
    pca = PCA(n_components=n_components)
    return pca.fit_transform(X)

# Add more feature selection methods as needed
