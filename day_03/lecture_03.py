# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 17:36:34 2022

@author: 오정민
"""

import numpy as np

# 1차원 배열
X = np.arange(1,11)
print(X)

# 1차원 배열을 2차원으로 수정
X = X.reshape(-1, 1)
print(X)

# 종속변수 - 연속된 수치형
y = np.arange(10, 101, 10)
print(y) 

# KNeighborsRegressor : 회귀 예측을 수행할 수 있는 클래스
# 최근점 이웃을 사용하여 회귀 예측을 수행하는 경우
# 분류 모델과 차이점은 다수결이 아닌 평균 값을
# 반환하는 점 입니다.


# 선형 방정식을 기반으로 회귀분석을 수행하는 머신러닝 클래스
from sklearn.neighbors import KNeighborsRegressor
model = KNeighborsRegressor(n_neighbors=1)

model.fit(X, y)

X_new = [[3.7]] # 3.7과 가장 가까운 수 40이 나오게 된다. 
pred = model.predict(X_new)
print(pred)

# 최근접 이웃 알고리즘의 단점
# -fit 메소드에서 입력된 X 데이터의 범위를 벗어나면
# 양 끝단의 값으로만 예측을 수행합니다.
# (학습 시 저장된 값 내에서만 예측이 가능하다)
X_new = [[57.7]] # 범위 값 밖이면 10,9가 나와 100 + 90 / 2 =95
pred = model.predict(X_new)
print(pred)

# 선형 방정식을 기반으로 회귀분석을 수행하는 머신러닝 클래스 
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)

X_new = [[3.7]]
pred = model.predict(X_new)
print(pred)

X_new = [[57.7]]
pred = model.predict(X_new)
print(pred)

X_new = [[-10.7]]
pred = model.predict(X_new)
print(pred)
