# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 12:08:43 2022

@author: 오정민
"""

import pandas as pd

from sklearn.datasets import load_iris

data = load_iris()

X = pd.DataFrame(data.data, columns = data.feature_names)
y = pd.Series(data.target)

print(X.info())
print(X.describe())

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

num_columns = X.columns

from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([('scaler',scaler,num_columns)]).fit(X)

print(X.describe())
print(ct.transform(X))
