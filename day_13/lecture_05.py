# -*- coding: utf-8 -*-
"""
Created on Fri May 27 15:06:57 2022

@author: 오정민
"""

# 병합 군집
# - 다수개의 소규모 군집을 생성 (랜덤하게)
# - 다수개의 소규모 군집을 취합해 하나로 병합 (인접한 위치의 군집사이에서 발생)
# - 원하는 개수의 군집으로 최정 처리를 완료


from sklearn.datasets import make_blobs # 군집데이터를 분석하는데 활용하는 함수
X, y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)

print(X[:10])
print(y[:10])

import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c='white', marker = 'o', edgecolor ='black', s=50)

plt.grid()
plt.show()

# 군집분석을 위한 클래스
# 병합 군집을 처리할 수 있는 클래스
 
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=3)

# 군집 분석 알고리즘인 AgglomerativeClustering
# fit_predict 메소드의 동작
# n_clusters에 정의된 개수만큼 소규모의 군집들을 계속해서 병합한 후 정의된
# 개수에 도달하면 해당정보 병합
y_cluster = ac.fit_predict(X)
print(y_cluster)


plt.scatter(X[y_cluster == 0, 0], X[y_cluster == 0, 1], s=50, c='lightgreen', marker ='v', label = 'Cluster 1')
plt.scatter(X[y_cluster == 1, 0], X[y_cluster == 1, 1], s=50, c='orange', marker ='o', label = 'Cluster 2')
plt.scatter(X[y_cluster == 2, 0], X[y_cluster == 2, 1], s=50, c='red', marker ='*', label = 'Cluster 3')

plt.legend()
plt.grid()
plt.show()
