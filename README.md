# Customer Segmentation Project

This project implements a customer segmentation analysis using K-means clustering algorithm. It aims to group customers based on their characteristics such as age, annual income, and spending score.

## Project Structure

- data/: Directory for storing the customer data CSV file
- notebooks/: Directory for exploratory data analysis and visualization notebooks
- src/: Directory containing the main Python script
- README.md: This file
- requirements.txt: List of required Python packages

## Setup

1. Clone this repository
2. Create a virtual environment and activate it
3. Install the required packages: `pip install -r requirements.txt`
4. Place your customer data CSV file in the `data/` directory and name it `customer_data.csv`

## Usage

Run the main script:

```
python src/customer_segmentation.py
```

This will perform the following steps:
1. Load the customer data
2. Preprocess the data
3. Perform K-means clustering
4. Visualize the clusters
5. Print summary statistics for each cluster

## Customization

You can adjust the number of clusters by modifying the `n_clusters` variable in the `main()` function of `customer_segmentation.py`.

## Cluster Plots

<img width="1817" height="874" alt="image" src="https://github.com/user-attachments/assets/3d0a2cc2-f88d-4638-b405-8166c969b5a6" />


## Contributing

Feel free to fork this project and submit pull requests with improvements or open issues if you find any problems.

## License


This project is open source and available under the MIT License.
