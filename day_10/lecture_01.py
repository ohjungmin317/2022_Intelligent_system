# -*- coding: utf-8 -*-
"""
Created on Fri May  6 15:07:51 2022

@author: 오정민
"""
import pandas as pd
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns= data.feature_names)
y = pd.Series(data.target)

print(X.info())

print(X.describe())


# 데이터 전처리
# 1. 문자열 
# - 결측 데이터
# - 라벨 인코딩 [이름들이 대/소 비교가 가능한것 -> 금메달 / 은메달 / 동메달 ]
# - 원핫 인코딩 [데이터에 대한 의미를 부여하는 것]

# 2. 수치형
# - 스케일링 
# - 이상치 제거(대체) [눈에 띄게 높거나 낮은것이 있는 것]

# - 전처리 클래스 로딩
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# - 전처리를 적용할 컬럼을 식별
num_columns = X.columns # 모든 x에 대한 columns를 가지고 온 것이다.

from sklearn.compose import ColumnTransformer # 전처리를 지원해주는 클래스
ct = ColumnTransformer([('scaler', scaler, num_columns)])
ct.fit(X)

print(X.head())
print(ct.transform(X)[:5])

