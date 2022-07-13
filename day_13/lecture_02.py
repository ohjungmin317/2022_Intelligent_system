# -*- coding: utf-8 -*-
"""
Created on Fri May 27 15:44:11 2022

@author: 오정민
"""

from sklearn.datasets import make_blobs # 군집데이터를 분석하는데 활용하는 함수
X, y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)

print(X[:10])
print(y[:10])

import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c='white', marker = 'o', edgecolor ='black', s=50)

plt.grid()
plt.show()
 
# 최근접 이웃 알고리즘
# 군집분석을 위한 클래스
# KMeans
# - 가장 많이 사용되는 군집분석 클래스
# - 알고리즘이 단순하고 변경에 용이
# (수정 사항의 반영이 손쉬움)

from sklearn.cluster import KMeans
km = KMeans(n_clusters=3, init='random', n_init=10, max_iter=300, random_state=0)

# fit 메소드 동작
# - n_clusters에 정의된 개수만큼 포인트를 지정하여 최적의 위치를 찾도록 검색하는 과정을 수행
km.fit(X)

# 군집의 결과를 생성하여 반환
y_cluster = km.predict(X)
print(y_cluster)

plt.scatter(X[y_cluster == 0, 0], X[y_cluster == 0, 1], s=50, c='lightgreen', marker ='s', label = 'Cluster 1')
plt.scatter(X[y_cluster == 1, 0], X[y_cluster == 1, 1], s=50, c='orange', marker ='o', label = 'Cluster 2')
plt.scatter(X[y_cluster == 2, 0], X[y_cluster == 2, 1], s=50, c='red', marker ='*', label = 'Cluster 3')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], s=50, c='lightgreen', marker ='s', label = 'Cluster 1')

plt.legend()
plt.grid()
plt.show()