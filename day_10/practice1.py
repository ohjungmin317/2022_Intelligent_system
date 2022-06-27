# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 11:47:30 2022

@author: 오정민
"""
import numpy as np
import pandas as pd
pd.options.display.max_rows=100
pd.options.display.max_columns=100

# 1. 데이터 적재
fname_input = './test1.csv'
data = pd.read_csv(fname_input)

print(data.head())
print(data.info())

data2 = data.drop(columns = 'Id')

print(data2.info())

X = data2.iloc[:,:-1]
y = data.SalePrice


print(data.isnull().sum())

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')

scaler = StandardScaler()

imputer_num = SimpleImputer(strategy='mean')

imputer_obj = SimpleImputer(strategy='most_frequent')

from sklearn.pipeline import Pipeline

pipe_num = Pipeline([('imputer_num',imputer_num),('scaler',scaler)])

pipe_obj = Pipeline([('imputer_obj',imputer_obj),('encoder',encoder)])

obj_columns = [cname for cname in X.columns if X[cname].dtype == 'object']
num_columns = [cname for cname in X.columns if X[cname].dtype in ['int64','float64']]


ct = ColumnTransformer([('pipe_num',pipe_num,num_columns),('pipe_obj',pipe_obj,obj_columns)]).fit(X)

X = ct.transform(X)

print(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=11)

print(X_train.shape, y_train.shape)

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

model_kn = KNeighborsRegressor().fit(X_train,y_train)
model_rf = RandomForestRegressor().fit(X_train,y_train)
model_gb = GradientBoostingRegressor().fit(X_train,y_train)

from sklearn.metrics import mean_absolute_error

score_kn = mean_absolute_error(y_test,model_kn.predict(X_test))
score_rf = mean_absolute_error(y_test, model_rf.predict(X_test))
score_gb = mean_absolute_error(y_test, model_gb.predict(X_test))

print(f'score_kn : {score_kn}')
print(f'score_rf : {score_rf}')
print(f'score_gb : {score_gb}')



