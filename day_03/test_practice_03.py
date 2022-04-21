# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 22:23:15 2022

@author: 오정민
"""

import numpy as np

X = np.arange(1,11)
print(X)

# 1차원 배열을 2차원으로 수정
X = X.reshape(-1, 1)
print(X)

# 종속변수 - 연속된 수치형
y = np.arange(10, 101, 10)
print(y)

# KNeighborsRegressor : 회귀 예측을 수행할 수 있는 클래스

# 최근접이웃을 사용하여 회귀예측을 수행하는경우 
# 분류모델과 차이점은 다수결이 아닌 평균의 값을 반환 
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=2)

model.fit(X,y)

X_new = [[3.7]] # n_neighbors = 1이면  3.7과 가장 가까운 것 4 -> 40
                # n_neighbors = 2이면  3.7에서 가장 인접한것 2개 3과 4 -> 30 + 40 / 2 = 35 
pred = model.predict(X_new)
print(pred)

# 최근접 이웃 알고리즘의 단점 (한계점)
# fit 메소드에서 입력된 X 데이터의 범위를 벗어나면
# 양 끝단의 값으로만 예측을 수행 [학습시시 저장된 값 내에서만 가능]
X_new = [[57.7]] 
pred = model.predict(X_new)
print(pred)


# X * w + b = y -> 가중치 값이 10이기 때문에 
# ex) 1 * w + b = 10 -> 1 * 10 = 10 
# 선형 방정식을 기반으로 회귀분석을 수행하는 머신러닝 클래스 
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X,y)

X_new = [[3.7]] # 값이 그대로 나옴 37
pred = model.predict(X_new)
print(pred)

X_new = [[57.7]]  # 값이 그래도 나옴 577
pred = model.predict(X_new)
print(pred)

X_new = [[-10.88]] # 값이 그대로 나옴 -108.8
pred = model.predict(X_new)
print(pred)