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


from sklearn.cluster import KMeans

values = []

for i in range(1,11):
    km = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, random_state=0)
    
    km.fit(X)
    
    values.append(km.inertia_)

print(values)

plt.plot(range(1,11), values, marker = 'o')
plt.xlabel('numbers of cluster')
plt.ylabel('inertia_')
plt.show()    
    
# 클래스터 내의 각 클래스의 SSE 값을 반환하는 inertia_ 속성 값