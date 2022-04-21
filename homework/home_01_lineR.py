# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 15:51:01 2022

@author: 오정민
"""
# 당뇨 수치 데이터(load_diabetes)를 사용하여 앙상블 기반의 회귀분석 모델을 구축하세요.
# 모델을 구축한 후 학습, 테스트 데이터에 대한 평균 절대 오차를 출력하여
# 모델의 적합성을 평가하세요.
#  - 앙상블 모델은 배깅, 그레디언트 부스팅을 사용하세요.

import pandas as pd

pd.options.display.max_columns=100

from sklearn.datasets import load_diabetes

data = load_diabetes()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)


X.info()
X.isnull().sum()
# Out[8]: 
# age    0
# sex    0
# bmi    0
# bp     0
# s1     0
# s2     0
# s3     0
# s4     0
# s5     0
# s6     0
# dtype: int64
# 결측데이터 이상 없음

X.describe()
# 스케일 차이 거의 없음

X.head()
X.tail()


y.head()
y.tail()
y.value_counts() # 중복요소수 세기
y.value_counts() / len(y) # 중복요소 수 세기 비율


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.3, random_state=1)

from sklearn.ensemble import BaggingRegressor

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error

from sklearn.linear_model import Ridge,Lasso



lr_base_estimator = LinearRegression(n_jobs = 1)

lr_model = BaggingRegressor(base_estimator=lr_base_estimator, n_estimators=100, 
                          max_samples=0.5,
                          max_features=0.5,
                          random_state=1)

ridge_base_estimator = Ridge(alpha=1.0, random_state=1 )

ridge_model = BaggingRegressor(base_estimator=ridge_base_estimator, n_estimators=100, 
                          max_samples=0.5,
                          max_features=0.5,
                          random_state=1)

lasso_base_estimator = Lasso(alpha=1.0, random_state=1 )

lasso_model = BaggingRegressor(base_estimator=lasso_base_estimator, n_estimators=100, 
                          max_samples=0.5,
                          max_features=0.5,
                          random_state=1)


lr_model.fit(X_train, y_train)

ridge_model.fit(X_train, y_train)

lasso_model.fit(X_train, y_train)


lr_train_pred = lr_model.predict(X_train)
lr_test_pred = lr_model.predict(X_test)

ridge_train_pred = ridge_model.predict(X_train)
ridge_test_pred = ridge_model.predict(X_test)

lasso_train_pred = ridge_model.predict(X_train)
lasso_test_pred = ridge_model.predict(X_test)


lr_score = lr_model.score(X_train, y_train)
print(f'lr_score (Train) : {lr_score}')

lr_score = lr_model.score(X_test, y_test)
print(f'lr_score (Test) : {lr_score}')


ridge_score = ridge_model.score(X_train, y_train)
print(f'ridge_score (Train) : {ridge_score}')

ridge_score = ridge_model.score(X_test, y_test)
print(f'ridge_score (Test) : {ridge_score}')

lasso_score = lasso_model.score(X_train, y_train)
print(f'lasso_score (Train) : {lasso_score}')

lasso_score = lasso_model.score(X_test, y_test)
print(f'lasso_score (Test) : {lasso_score}')


lr_train_mae = mean_absolute_error(y_train, lr_train_pred)

print(f'Train_Mae : {lr_train_mae}')

lr_test_mae = mean_absolute_error(y_test, lr_test_pred)

print(f'Test_Mae : {lr_test_mae}')


ridge_train_mae = mean_absolute_error(y_train, ridge_train_pred)

print(f'Train_Mae : {ridge_train_mae}')

ridge_test_mae = mean_absolute_error(y_test, ridge_test_pred)

print(f'Test_Mae : {ridge_test_mae}')


lasso_train_mae = mean_absolute_error(y_train, lasso_train_pred)

print(f'Train_Mae : {lasso_train_mae}')

lasso_test_mae = mean_absolute_error(y_test, lasso_test_pred)

print(f'Test_Mae : {lasso_test_mae}')


