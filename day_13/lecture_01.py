# -*- coding: utf-8 -*-
"""
Created on Fri May 27 15:06:57 2022

@author: 오정민
"""

# 비지도 학습 / 스태킹 학습 

# 비지도 학습
# - 지도 학습 / 비지도 학습 / [준지도 학습 / 강화학습]
# - 지도학습과 다르게 종속변수(정답데이터)가 제공되지 않는 데이터에 대한 학습을 처리하는 기법
# - 비지도학습의 결과는 주관적인 판단으로 처리

# 비지도 학습의 카테고리
# - 차원 축소 ( 시각화를 위해서 사용되는 경우가 많음 / 소셜 네트워크 분석 / 이미지 분석 / RGB 컬러 )
# 데이터에 포함된 특성 중 유의마한 값을 추출
# 데이터에 포함된 특성을 투영하여 대표하는 값을 반환
# 가지고 있는 데이터중 대표하는 데이터를 뽑아내서 출력해줄 수 있다. [시각화 방법에 많이 사용]
 
# - 군집 분석
# 데이터(샘플)의 유사성을 비교하여 동일한 특성으로 구성된 데이터(샘플)들을 하나의 군집으로 처리하는 기법

# 비지도 학습 시 유의사항
# - 비지도 학습의 결과는 100% 신뢰할 수 없음
# - 매번 실행될 때마다 결과는 변경될 수 있음

# 군집 분석을 활용하여 데이터의 클러스터링 처리 과정을 확인
# - 지도학습에서 군집 분석의 결과 활용하는 방법
# - 군집 처리를 할때에도 scale 처리를 해줘야한다

from sklearn.datasets import make_blobs # 군집데이터를 분석하는데 활용하는 데이터 ( 분류용 가상 데이터 )
X, y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)

print(X[:10])
print(y[:10])

import matplotlib.pyplot as plt
plt.scatter(X[:, 0], X[:, 1], c='white', marker = 'o', edgecolor ='black', s=50)

plt.grid()
plt.show()
 
from sklearn.cluster import KMeans
km = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, random_state=0)

# fit 메소드 동작
# - n_clusters에 정의된 개수만큼 포인트를 지정하여 최적의 위치를 찾도록 검색하는 과정을 수행
km.fit(X)

# 군집의 결과를 생성하여 반환
y_cluster = km.predict(X)
print(y_cluster)

plt.scatter(X[y_cluster == 0, 0], X[y_cluster == 0, 1], s=50, c='lightgreen', marker ='v', label = 'Cluster 1')
plt.scatter(X[y_cluster == 1, 0], X[y_cluster == 1, 1], s=50, c='orange', marker ='o', label = 'Cluster 2')
plt.scatter(X[y_cluster == 2, 0], X[y_cluster == 2, 1], s=50, c='red', marker ='*', label = 'Cluster 3')
plt.scatter(X[y_cluster == 3, 0], X[y_cluster == 3, 1], s=50, c='yellow', marker ='s', label = 'Cluster 4')
plt.scatter(X[y_cluster == 4, 0], X[y_cluster == 4, 1], s=50, c='blue', marker ='s', label = 'Cluster 5')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], s=50, c='pink', marker ='s', label = 'Center')

plt.legend()
plt.grid()
plt.show()