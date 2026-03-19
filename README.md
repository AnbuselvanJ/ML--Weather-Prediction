# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the required libraries and load the dataset containing customer details.
2. Select relevant features (Annual Income and Spending Score) for clustering.
3. Initialize the K-Means model with a predefined number of clusters (k = 5) and fit the model to the selected data.
6. Assign cluster labels to each data point and append the cluster information to the dataset.
5. Visualize the clusters along with their centroids using a scatter plot to analyze customer segmentation.

## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: Anbuselvan J
RegisterNumber: 212225230015
*/
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data = pd.read_csv("Mall_Customers.csv")
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]
print(data.head())
kmeans = KMeans(n_clusters=5, random_state=42)
y_kmeans = kmeans.fit_predict(X)


data['Cluster'] = y_kmeans

print("\nClustered Data:")
print(data.head())


plt.figure()
plt.scatter(X[y_kmeans == 0]['Annual Income (k$)'], 
            X[y_kmeans == 0]['Spending Score (1-100)'], label='Cluster 0')

plt.scatter(X[y_kmeans == 1]['Annual Income (k$)'], 
            X[y_kmeans == 1]['Spending Score (1-100)'], label='Cluster 1')

plt.scatter(X[y_kmeans == 2]['Annual Income (k$)'], 
            X[y_kmeans == 2]['Spending Score (1-100)'], label='Cluster 2')

plt.scatter(X[y_kmeans == 3]['Annual Income (k$)'], 
            X[y_kmeans == 3]['Spending Score (1-100)'], label='Cluster 3')

plt.scatter(X[y_kmeans == 4]['Annual Income (k$)'], 
            X[y_kmeans == 4]['Spending Score (1-100)'], label='Cluster 4')

# Plot centroids
plt.scatter(kmeans.cluster_centers_[:,0], 
            kmeans.cluster_centers_[:,1], 
            s=200, label='Centroids')

plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()
```

## Output:
<img width="1051" height="206" alt="Screenshot 2026-03-19 152823" src="https://github.com/user-attachments/assets/b40adf93-9a90-4da0-83cd-1fd03515e7f3" />
<img width="1059" height="467" alt="Screenshot 2026-03-19 152831" src="https://github.com/user-attachments/assets/d57bc25c-a8b1-4a46-85f3-f4994cfde07a" />
<img width="1051" height="718" alt="Screenshot 2026-03-19 152839" src="https://github.com/user-attachments/assets/b4bf2cd8-82c4-4aa5-aece-ad590cdaa91d" />


## Result:
