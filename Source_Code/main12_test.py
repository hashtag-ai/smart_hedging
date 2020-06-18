import os
import matplotlib.pyplot as plt
import pandas as pd
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

wdir = os.getcwd()
datadir = os.path.join(wdir,"data")

customer_data_filename = "/hierarchical-clustering-with-python-and-scikit-learn-shopping-data.csv"
full_path_customer_data = datadir+customer_data_filename

customer_data = pd.read_csv(full_path_customer_data)

customer_data.shape

customer_data.head()

data = customer_data.iloc[:, 3:5].values

plt.figure(figsize=(10, 7))
plt.title("Customer Dendrograms")
dend = shc.dendrogram(shc.linkage(data, method='ward'))

cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
cluster.fit_predict(data)

plt.figure(figsize=(10, 7))
plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')

