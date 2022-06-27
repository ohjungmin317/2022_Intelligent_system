# -*- coding: utf-8 -*-
"""
Created on Fri May  6 15:36:47 2022

@author: 오정민
"""

import pandas as pd 

X = pd.DataFrame()

print(X) # 비어있는 데이터 출력

X['gender'] = ['F','M','F','F','M']
print(X)

X['age'] = [15, 35, 25, 37, 55]
print(X)

# 데이터 전처리
# 1. 문자열 
# - 결측 데이터
# - 라벨 인코딩 [이름들이 대/소 비교가 가능한것 -> 금메달 / 은메달 / 동메달 ]
# - 원핫 인코딩 [데이터에 대한 의미를 부여하는 것]

# 2. 수치형
# - 스케일링 
# - 이상치 제거(대체) [눈에 띄게 높거나 낮은것이 있는 것]


from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
 
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

scaler = MinMaxScaler()

from sklearn.compose import ColumnTransformer

obj_columns = ['gender']
num_columns = ['age']

ct = ColumnTransformer([('scaler',scaler,num_columns),('encoder',encoder,obj_columns)])

ct.fit(X)

print(X)
print(ct.transform(X))
