# Kaggle Competition - House Price Prediction

This notebook contains the code to predict house prices based on various features. This is a part of Kaggle competition and the source data is downloaded from Kaggle site. The data is in CSV format and pre-split into - train.csv and test.csv

Competition Link:
https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview

March 25, 2023:
The notebook is in progress state. This is an ongoing work and will continue to be updated.

Steps Performed before Data Modeling -
1. Package imports
2. File read from google drive location
3. Files preprocessed - imputing missing data, changing non-numerical categorical values to numerical categorical values, using StandardScalar() on the data. This process is done for both train.csv and test.csv.
4. Correlation Matrix and Heatmap
5. Principle Component Analysis

Data Modeling -
1. Random Forest - Random Forest model is generated using 62 features out of 80 features from the original data. This model is generated without any cross validation.

 
