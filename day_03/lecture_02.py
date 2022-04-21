# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 16:42:12 2022

@author: 오정민
"""

import pandas as pd

X = pd.DataFrame()
print(X)

X['rate'] = [0.3,0.8,0.9999]
print(X)

X['Price'] = [10000, 5000, 9500]
print(X)

y = pd.Series([0,1,0])


# scale 전처리 과정을 수행 
# ~ price 컬럼의 값을 rate 컬럼의 값과 동일한 범위를 가지도록 데이터 수정
# (데이터의 값은 수정되지만 원본 값에서 가지는 상대적인 크기는 유지)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# MinMaxScaler 전처리 수행과정
# 1, 각 컬럼 별 최소 / 최대값을 추출
# 2, 각 컬럼 별 아래의 연산을 수행하여 값을 대체
# - (원본 값 - 최소 값) / (최대값 - 최소값)
#  결과적으로 모든 컬럼의 값은 최대 값이 1로 변환 되고 
#  최소값은 0으로 변환이 된다.

scaler.fit(X)

# 스케일 처리 수행 코드 (실제 변환되는 코드)
X = scaler.transform(X)
print(X)



from sklearn.neighbors import KNeighborsClassifier

# 가자 인접한 1개의 데이터를 기준으로 판단 
model = KNeighborsClassifier(n_neighbors=1)

model.fit(X,y)

# 예측할 데이터 -> 2차원 X이기 때문에 
X_new = [[0.81, 7000]]

# 예측할 데이터의 스케일 처리를 수행 
X_new = scaler.transform(X_new)

# 예측 수행
pred = model.predict(X_new)
print(pred)


# 학습에 사용된 X 데이터 
#       rate  Price
# 0  0.3000  10000
# 1  0.8000   5000
# 2  0.9999   9500

# 학습에 사용된 y 데이터
# [0,1,0]

# 예측할 데이터
# X_new = [[0.81, 7000]]

# 유클리드 거리 (두 값의 차이를 제곱한 후, 제곱근 값을 취함 )
# 동일 특성 사이에서 차이를 계산
# - (rate - new_rate) ** 2  + - (price - price) ** 2
# (0.3 - 0.81) + (10000 - 7000) = 3000.51 [0]
# (0.8 - 0.81) + (5000 - 7000) = 2000.01 (가장 인접한 값)[1] --> 값이 1이 나옴
# (0.9 - 0.81) + (9500 - 7000) = 2500.09[0]


# 최근접 이웃 알고리즘의 학습 및 예측 방법
# ~ 학습 : fit 메소드에 입력된 데이터를 단순 저장
# ~ 예측 : fit 메소드에 의해서 저장된 데이터와
#       예측하고자 하는 신규 데이터와의 유클리드 거리를 계산
#       가장 인접한 n_neighbors 개수를 사용하여 이웃을 추출
#       추출된 이웃의 y를 사용하여 다수결의 과정을 수행 