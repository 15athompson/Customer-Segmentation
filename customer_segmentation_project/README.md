# Customer Segmentation Project

## Overview
This project focuses on sophisticated customer segmentation through the application of various advanced clustering techniques. By identifying distinct customer groups based on intricate purchasing behaviors, this initiative enables the development of targeted marketing strategies to enhance customer engagement and satisfaction.

## Project Structure
```
customer_segmentation_project/
|
+-- data/
|   +-- sample_customer_data.csv
|
+-- src/
|   +-- customer_segmentation.py
|   +-- advanced_clustering.py
|   +-- feature_selection.py
|   +-- dashboard.py
|
+-- notebooks/
|   +-- exploratory_analysis.ipynb
|
+-- results/
|
+-- requirements.txt
```

## Setup and Installation
1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - On Windows: `venv\Scripts\activate`
   - On macOS and Linux: `source venv/bin/activate`
4. Install the required packages: `pip install -r requirements.txt`

## Usage
1. Place your customer data in the `data/` directory
2. Run the main script: `python src/customer_segmentation.py`
3. Check the `results/` directory for the output
4. Explore the data and visualizations in `notebooks/exploratory_analysis.ipynb`
5. To run the dashboard:
   - Ensure you have installed all requirements
   - Run `python src/dashboard.py`
   - Open a web browser and navigate to `http://127.0.0.1:8050/` to view the dashboard

## Features
- Data preprocessing and feature engineering
- Implementation of multiple clustering algorithms (KMeans, DBSCAN)
- Dimensionality reduction using PCA for visualization
- Exploratory data analysis and cluster visualization
- Cluster profiling for marketing insights
- Advanced clustering techniques:
  - Hierarchical clustering
  - Gaussian Mixture Models
  - Bayesian Gaussian Mixture Models
- Feature selection methods:
  - SelectKBest
  - Mutual Information
  - Principal Component Analysis (PCA)
- Real-time dashboard for customer insights:
  - Interactive visualizations of customer segments
  - Dynamic updates of customer data
  - Spending distribution analysis by segment

## Contributing
Contributions to improve the project are welcome. Please follow these steps:
1. Fork the repository
2. Create a new branch
3. Make your changes and commit them
4. Push to your fork and submit a pull request

## License
This project is licensed under the MIT License.
