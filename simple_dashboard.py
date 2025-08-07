#!/usr/bin/env python3
"""
Simple Customer Segmentation Dashboard
A lightweight dashboard using matplotlib for visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os
import sys

def load_customer_data():
    """Load customer data from CSV file"""
    try:
        # Try to find the customer data file
        possible_paths = [
            'customer_data.csv',
            'Customer-Segmentation/customer_data.csv',
            os.path.join('Customer-Segmentation', 'customer_data.csv')
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                print(f"Loading data from: {path}")
                return pd.read_csv(path)
        
        print("Customer data file not found. Creating sample data...")
        # Create sample data if file not found
        np.random.seed(42)
        return pd.DataFrame({
            'CustomerID': range(1, 101),
            'Age': np.random.randint(18, 70, 100),
            'Annual Income (k$)': np.random.randint(15, 140, 100),
            'Spending Score (1-100)': np.random.randint(1, 100, 100)
        })
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def perform_clustering(df):
    """Perform K-means clustering on the data"""
    # Prepare features for clustering
    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
    X = df[features].values
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    # Add segment labels
    segment_labels = {0: 'Budget Conscious', 1: 'High Value', 2: 'Young Spenders', 3: 'Conservative'}
    df['Segment'] = df['Cluster'].map(segment_labels)
    
    return df, kmeans, scaler

def create_dashboard_plots(df):
    """Create dashboard visualizations"""
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Customer Segmentation Dashboard', fontsize=16, fontweight='bold')
    
    # Plot 1: Segment Distribution (Pie Chart)
    segment_counts = df['Segment'].value_counts()
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']
    axes[0, 0].pie(segment_counts.values, labels=segment_counts.index, autopct='%1.1f%%', 
                   colors=colors, startangle=90)
    axes[0, 0].set_title('Customer Segment Distribution')
    
    # Plot 2: Age vs Income Scatter Plot
    for i, segment in enumerate(df['Segment'].unique()):
        segment_data = df[df['Segment'] == segment]
        axes[0, 1].scatter(segment_data['Age'], segment_data['Annual Income (k$)'], 
                          label=segment, alpha=0.7, color=colors[i])
    axes[0, 1].set_xlabel('Age')
    axes[0, 1].set_ylabel('Annual Income (k$)')
    axes[0, 1].set_title('Age vs Annual Income by Segment')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Spending Score Distribution
    df.boxplot(column='Spending Score (1-100)', by='Segment', ax=axes[1, 0])
    axes[1, 0].set_title('Spending Score Distribution by Segment')
    axes[1, 0].set_xlabel('Segment')
    axes[1, 0].set_ylabel('Spending Score (1-100)')
    plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 4: Income vs Spending Score
    for i, segment in enumerate(df['Segment'].unique()):
        segment_data = df[df['Segment'] == segment]
        axes[1, 1].scatter(segment_data['Annual Income (k$)'], segment_data['Spending Score (1-100)'], 
                          label=segment, alpha=0.7, color=colors[i])
    axes[1, 1].set_xlabel('Annual Income (k$)')
    axes[1, 1].set_ylabel('Spending Score (1-100)')
    axes[1, 1].set_title('Income vs Spending Score by Segment')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def print_segment_summary(df):
    """Print summary statistics for each segment"""
    print("\n" + "="*60)
    print("CUSTOMER SEGMENT ANALYSIS SUMMARY")
    print("="*60)
    
    for segment in df['Segment'].unique():
        segment_data = df[df['Segment'] == segment]
        print(f"\n{segment.upper()} SEGMENT:")
        print(f"  Count: {len(segment_data)} customers ({len(segment_data)/len(df)*100:.1f}%)")
        print(f"  Average Age: {segment_data['Age'].mean():.1f} years")
        print(f"  Average Income: ${segment_data['Annual Income (k$)'].mean():.1f}k")
        print(f"  Average Spending Score: {segment_data['Spending Score (1-100)'].mean():.1f}/100")

def main():
    """Main function to run the dashboard"""
    print("Starting Customer Segmentation Dashboard...")
    
    # Load data
    df = load_customer_data()
    if df is None:
        print("Failed to load data. Exiting...")
        return
    
    print(f"Loaded {len(df)} customer records")
    
    # Perform clustering
    df_clustered, kmeans, scaler = perform_clustering(df)
    print("Clustering completed successfully!")
    
    # Create and display plots
    fig = create_dashboard_plots(df_clustered)
    
    # Print segment summary
    print_segment_summary(df_clustered)
    
    # Save clustered data first
    output_csv = 'clustered_customers.csv'
    df_clustered.to_csv(output_csv, index=False)
    print(f"Clustered data saved as: {output_csv}")

    # Save the plot
    output_file = 'customer_segmentation_dashboard.png'
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Dashboard saved as: {output_file}")

    # Show the plot (this blocks execution until window is closed)
    print("\nDisplaying dashboard... Close the plot window to continue.")
    plt.show()

if __name__ == "__main__":
    main()
