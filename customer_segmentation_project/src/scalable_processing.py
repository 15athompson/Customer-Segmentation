import dask.dataframe as dd
from dask_ml.cluster import KMeans

def process_large_dataset(file_path, n_clusters=5):
    """
    Process a large dataset using Dask for distributed computing.
    """
    # Read data using Dask
    ddf = dd.read_csv(file_path)
    
    # Perform clustering using Dask-ML
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(ddf)
    
    # Add cluster labels to the dataframe
    ddf['cluster'] = labels
    
    return ddf
