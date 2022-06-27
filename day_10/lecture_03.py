# -*- coding: utf-8 -*-
"""
Created on Fri May  6 15:49:59 2022

@author: 오정민
"""

import numpy as np
import pandas as pd

X = pd.DataFrame()

print(X)
print(X.info())

X['gender'] = ['F','M','F','F', None]
print(X)

X['age'] = [15, None, 25, 37, 55]
print(X)

print(X.info())
print(X.isnull().sum())

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer #사용자가 생각하는 결측데이터를 적으면 된다. 

encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

scaler = MinMaxScaler()


imputer_num = SimpleImputer(
    missing_values=np.nan, 
    strategy='mean')

imputer_obj = SimpleImputer(
    missing_values=None, 
    strategy='most_frequent')
# 최빈도 값을 구해주는 것이기 때문에 most_frequent는 F가 된다.
#   gender   age
# 0      F  15.0
# 1      M   NaN
# 2      F  25.0
# 3      F  37.0
# 4   None  55.0
# F-> 3이고 M이 1이기 때문에 

# pipeline을 통해서 하는 법
from sklearn.pipeline import Pipeline
num_pipe = Pipeline([('imputer_num',imputer_num),('scaler',scaler)]) # 수치형 관련된 파이프라인

obj_pipe = Pipeline([('imputer_obj',imputer_obj),('encoder',encoder)]) # 문자열 관련된 파이프라인
 
# pipeline 왜 사용? : 전처리를 해야하면 먼저해야하는 것이 있고 나중에 해야하는 것이 있다
# 먼저한 결과를 뒤에있는 놈이 받아야하는 경우가 있다.
# -> pipeline


# ColumnTransformer하는법 
from sklearn.compose import ColumnTransformer

obj_columns = ['gender']
num_columns = ['age']


# 결측데이터를 먼저 처리를 해주고 scaler처리를 해주면 된다. 
# scaler 처리를 해주려면 숫자가 들어가 있어야 하기 때문에 먼저 결측처리를 해주는 것이 좋다.

ct = ColumnTransformer(
    [('num_pipe',num_pipe,num_columns),
     ('obj_pipe',obj_pipe,obj_columns)])


ct.fit(X)

print(X)
print(ct.transform(X))
