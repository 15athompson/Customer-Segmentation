import pandas as pd
import requests

def fetch_external_data(api_key, customer_ids):
    """
    Fetch external data for given customer IDs.
    This is a placeholder function. In a real-world scenario,
    you would integrate with actual external APIs.
    """
    # Placeholder for external API call
    external_data = pd.DataFrame({
        'customer_id': customer_ids,
        'credit_score': np.random.randint(300, 850, len(customer_ids)),
        'social_media_score': np.random.randint(1, 100, len(customer_ids))
    })
    return external_data

def enrich_customer_profiles(customer_data, external_data):
    """
    Merge customer data with external data to create richer profiles.
    """
    return pd.merge(customer_data, external_data, on='customer_id', how='left')
