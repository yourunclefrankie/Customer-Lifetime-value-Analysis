Predicting Customer Lifetime Value Using RFMT
Description
This project aims to predict Customer Lifetime Value (CLV) using an RFMT model, which stands for 
Recency, Frequency, Monetary, and Tenure. The goal is to leverage these customer behavior metrics 
and apply machine learning models, specifically Random Forest Regressor and Gradient Boosting 
Regressor, to predict CLV. The project also includes K-Means clustering to segment customers based 
on RFMT values.
 
Features
RFMT Segmentation: Extract and analyze customer data based on Recency, Frequency, Monetary, 
and Tenure metrics.
Machine Learning Models: Train predictive models like Random Forest and Gradient Boosting for 
CLV estimation.
K-Means Clustering: Perform customer segmentation for better understanding of customer groups.
Performance Evaluation: Evaluate models using RMSE, MAE, and R-squared metrics.
 
RFMT Metrics
Recency: How recently the customer made a purchase.
Frequency: How often the customer makes a purchase.
Monetary: The total value of purchases made by the customer.
Tenure: The length of time the customer has been with the business.
 
Requirements
To run this project, ensure you have the following dependencies installed:
Python 3.x
Libraries:
numpy
pandas
scikit-learn
matplotlib
Install the required libraries by running:
bash
Copy code
pip install numpy pandas scikit-learn matplotlib
 
Project Steps
1. Import Libraries
python
Copy code
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
These libraries are essential for data manipulation, model training, evaluation, and visualization.
2. Loading and Preparing Data
python
Copy code
df = pd.read_csv('customer_data.csv')
X = df[['Recency', 'Frequency', 'Monetary', 'Tenure']]  # RFMT features
y = df['Customer_Lifetime_Value']  # Target variable
The dataset is loaded from a CSV file. The feature set X includes RFMT metrics, and the target 
variable y is the Customer Lifetime Value (CLV).
3. Splitting the Data
python
Copy code
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
The dataset is split into training (80%) and testing (20%) sets.
4. Training Models
(i) Random Forest Regressor
python
Copy code
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
The Random Forest model is trained using the training data.
(ii) Gradient Boosting Regressor
python
Copy code
gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)
The Gradient Boosting model is trained using 100 estimators and a learning rate of 0.1.
5. K-Means Clustering
python
Copy code
wcss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
K-Means clustering is used to identify customer segments based on their RFMT values, by 
experimenting with 1 to 9 clusters.
6. Evaluating Models
python
Copy code
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"RMSE: {rmse:.2f}, MAE: {mae:.2f}, R-squared: {r2:.2f}")
This function evaluates the performance of a trained model using the following metrics:
RMSE (Root Mean Squared Error): Measures prediction error.
MAE (Mean Absolute Error): Measures the average magnitude of errors.
R-squared: Indicates the proportion of variance explained by the model.
7. Making Predictions
python
Copy code
y_pred_rf = rf_model.predict(X_test)
y_pred_gb = gb_model.predict(X_test)
Predictions are made on the test data using the trained Random Forest and Gradient Boosting 
models.
8. Visualizing the Elbow Method
The elbow method helps determine the optimal number of clusters for customer segmentation:
python
Copy code
plt.plot(range(1, 10), wcss)
plt.title('Elbow Method for K-Means')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()
This plot is used to visually select the optimal number of clusters based on where the WCSS starts to 
decrease at a slower rate.
 
Running the Project
Install Dependencies: Run the following command to install necessary libraries:
bash
Copy code
pip install -r requirements.txt
Run the Project: Run the Python script that contains the training and evaluation code:
bash
Copy code
python main.py
Outputs:
Predicted Customer Lifetime Value for the test set.
Clustering visualization (Elbow method).
Performance metrics: RMSE, MAE, R-squared.

