import unittest
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from test import (
    load_data,
    preprocess_data,
    perform_kmeans,
    evaluate_clusters,
    find_optimal_clusters,
    profile_clusters
)

class TestCustomerSegmentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load sample data for testing
        cls.sample_data = pd.DataFrame({
            'Age': [25, 30, 35, 40, 45],
            'Annual Income (k$)': [30, 50, 70, 90, 110],
            'Spending Score (1-100)': [20, 40, 60, 80, 100]
        })

    def test_load_data(self):
        # Test load_data function (this might need to be adjusted based on your actual data file)
        df = load_data("path/to/test_data.csv")
        self.assertIsInstance(df, pd.DataFrame)
        self.assertGreater(len(df), 0)

    def test_preprocess_data(self):
        X_preprocessed, features = preprocess_data(self.sample_data)
        self.assertIsInstance(X_preprocessed, np.ndarray)
        self.assertEqual(X_preprocessed.shape[1], len(features))

    def test_perform_kmeans(self):
        X_preprocessed, _ = preprocess_data(self.sample_data)
        kmeans = perform_kmeans(X_preprocessed, n_clusters=3)
        self.assertIsInstance(kmeans, KMeans)
        self.assertEqual(len(np.unique(kmeans.labels_)), 3)

    def test_evaluate_clusters(self):
        X_preprocessed, _ = preprocess_data(self.sample_data)
        kmeans = perform_kmeans(X_preprocessed, n_clusters=3)
        score = evaluate_clusters(X_preprocessed, kmeans.labels_)
        self.assertIsInstance(score, float)
        self.assertGreaterEqual(score, -1)
        self.assertLessEqual(score, 1)

    def test_find_optimal_clusters(self):
        X_preprocessed, _ = preprocess_data(self.sample_data)
        optimal_clusters = find_optimal_clusters(X_preprocessed, max_clusters=5)
        self.assertIsInstance(optimal_clusters, int)
        self.assertGreaterEqual(optimal_clusters, 2)
        self.assertLessEqual(optimal_clusters, 5)

    def test_profile_clusters(self):
        X_preprocessed, _ = preprocess_data(self.sample_data)
        kmeans = perform_kmeans(X_preprocessed, n_clusters=3)
        self.sample_data['Cluster'] = kmeans.labels_
        profile = profile_clusters(self.sample_data)
        self.assertIsInstance(profile, pd.DataFrame)
        self.assertEqual(len(profile), 3)  # 3 clusters

if __name__ == '__main__':
    unittest.main()