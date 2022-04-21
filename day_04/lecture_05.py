# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 15:33:22 2022

@author: 오정민
"""

import pandas as pd

from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()

X = pd.DataFrame(data.data, columns = data.feature_names)

y = pd.Series(data.target)

print(X.info())
print(X.isnull().sum())

pd.options.display.max_columns = 100
print(X.describe(include='all'))

print(y.head())
print(y.tail())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.3,
                                                    random_state=1)
print(X_train.shape, X_test.shape)

print(len(y_train), len(y_test))



from sklearn.linear_model import LinearRegression

# 선형 모델에 제약조건 (L2, L1 제약조건)을 추가한 클래스
from sklearn.linear_model import Ridge, Lasso

# 머신러닝 객체의 생성
lr_model = LinearRegression(n_jobs=-1)

# Ridge, Lasso 클래스의 하이퍼 파라메터 alpha
# alpha의 값이 커질수록 제약을 크게 설정
# (alpha의 값이 커질수록 모든 특성들의 가중치의 값은 0주변으로 위치함)
# alpha의 값이 작아질수록 제약이 약해짐
# (alpha의 값이 작아질수록 모든 특성들의 가중치의 값은 0에서 멀어짐)
#alpha의 값이 작아질수록 LinearRegression 클래스와 동일해짐
# -> 제약조건을 주면 줄 수록 학습하는데있어 방해가 된다.

ridge_model = Ridge(alpha=1.0, random_state=1)
lasso_model = Lasso(alpha=0.001, random_state=1)


#학습
lr_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)

# Lasso 클래스를 사용하여 모델을 구축하면 대다수의 특성 가중치는 0으로 수렴(alpha 값에 따라 조정)
print(lasso_model.coef_)
 
# 학습 
score = lr_model.score(X_train, y_train)
print(f'Train (LR) : {score}')
score = ridge_model.score(X_train, y_train)
print(f'Train (Ridge) : {score}')
score = lasso_model.score(X_train, y_train)
print(f'Train (Lasso) : {score}')

# 테스트
score = lr_model.score(X_test, y_test)
print(f'Test (LR) : {score}')
score = ridge_model.score(X_test, y_test)
print(f'Test (Ridge) : {score}')
score = lasso_model.score(X_test, y_test)
print(f'Test (Lasso) : {score}')







        

