# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:54:47 2022

@author: 오정민
"""
from sklearn.datasets import make_moons

X,y = make_moons(n_samples=200, noise=0.05, random_state=0)

print(X[:10])
print(y[:10])

import matplotlib.pyplot as plt

plt.scatter(X[:,0], X[:,1])
plt.show()


# KMeans 사용 실패
from sklearn.cluster import KMeans
km = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, random_state=0)

km.fit(X)

y_cluster = km.predict(X)

plt.scatter(X[y_cluster == 0, 0], X[y_cluster == 0, 1], s=50, c='lightgreen', marker ='v', label = 'Cluster 1')
plt.scatter(X[y_cluster == 1, 0], X[y_cluster == 1, 1], s=50, c='orange', marker ='o', label = 'Cluster 2')
plt.scatter(X[y_cluster == 2, 0], X[y_cluster == 2, 1], s=50, c='red', marker ='*', label = 'Cluster 3')

plt.legend()
plt.grid()
plt.show()


# AgglomerativeClustering 사용

from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=2)

ac.fit(X)

y_cluster = km.predict(X)

plt.scatter(X[y_cluster == 0, 0], X[y_cluster == 0, 1], s=50, c='lightgreen', marker ='v', label = 'Cluster 1')
plt.scatter(X[y_cluster == 1, 0], X[y_cluster == 1, 1], s=50, c='orange', marker ='o', label = 'Cluster 2')
plt.scatter(X[y_cluster == 2, 0], X[y_cluster == 2, 1], s=50, c='red', marker ='*', label = 'Cluster 3')

plt.legend()
plt.grid()
plt.show()



