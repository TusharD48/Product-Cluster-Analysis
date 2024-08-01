# **Product Cluster Analysis**
# Overview
This project focuses on the clustering of products based on various attributes to identify patterns and group similar products together. Clustering can help in inventory management, marketing strategies, and enhancing the overall customer experience by understanding product affinities.

# Table of Contents
- Project Description
- Dataset
- Prerequisites
- Installation
- Usage
- Methods
- Results
- Contributing
- License
- Contact
# Project Description
The primary objective of this project is to apply clustering techniques on a product dataset to identify distinct groups. The analysis includes preprocessing of data, application of clustering algorithms, and evaluation of the results to derive meaningful insights.

# Dataset
The dataset contains the following columns:

- YEAR
- MONTH
- SUPPLIER
- ITEM CODE
- ITEM DESCRIPTION
- ITEM TYPE
- RETAIL SALES
- RETAIL TRANSFERS
- WAREHOUSE SALES
  
# Prerequisites
- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
  
# Installation
Clone the repository:
- https://github.com/TusharD48/Product-Cluster-Analysis.git
  
# Methods
- Data Preprocessing: Handling missing values, encoding categorical variables, and normalizing numerical features.
- Clustering Algorithms: K-Means, Mini batch K-Means
- Evaluation Metrics: Silhouette Score, Davies-Bouldin Index.
- Visualization: Cluster heatmaps, scatter plots, and distribution plots.
- Results: The optimal number of clusters was determined to be 5 based on the Silhouette Score. The clustering results reveal distinct groups of products with similar characteristics. Detailed results and visualizations can be found in the https://github.com/TusharD48/Product-Cluster-Analysis/blob/main/Product_cluster_analysis_main.ipynb/ directory.
