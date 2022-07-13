# -*- coding: utf-8 -*-
"""
Created on Thu Jun 16 00:31:52 2022

@author: 오정민
"""

# 스태킹 모델의 구축
# - 앙상블 : 다수개의 머신러닝 모델의 예측 값을 취합하여 평균, 다수결의 원칙으로 예측하는 모델
# - 앙상블을 사용하는 이유?
# - 일반화의 성능을 극대화 하기 위해서 ( 예측 성능의 분산을 감소시킬 수 있으므로 )

# 앙상블은 new data가 들어오게 되면 결과에 따라서 다수결을 따른다.
# 스태킹 :  각 다수개의 에측한 모델들이 학습한 결과를 취합하여 그것을 기초로 하여 새로운 머신러닝으로 예측하는 것

import pandas as pd
pd.options.display.max_columns=100
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X = pd.DataFrame(data.data, 
                 columns=data.feature_names)
y = pd.Series(data.target, name='target')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, stratify=y, random_state=1)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler().fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 앙상블 구현
# 다수개의 모델을 사용하여 학습 및 예측을 진행한것
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier



lr = LogisticRegression(C=1.0, random_state=0, class_weight='balanced').fit(X_train_scaled, y_train)
kn = KNeighborsClassifier().fit(X_train_scaled, y_train)
df = DecisionTreeClassifier(random_state=0).fit(X_train_scaled, y_train) 


v_score = lr.score(X_train_scaled, y_train)
print(f'학습 lr : {v_score}')

v_score = kn.score(X_train_scaled, y_train)
print(f'학습 kn : {v_score}')

v_score = df.score(X_train_scaled, y_train)
print(f'학습 df : {v_score}')


v_score = lr.score(X_test_scaled, y_test)
print(f'테스트 lr : {v_score}')

v_score = kn.score(X_test_scaled, y_test)
print(f'테스트 kn : {v_score}') 

v_score = df.score(X_test_scaled, y_test)
print(f'테스트 df : {v_score}')

# 스태킹 구현
# 1, 앙상블을 구현하고 있는 각 머신러닝 모델의 예측 결과를 취함

pred_lr = lr.predict(X_train_scaled)
pred_kn = kn.predict(X_train_scaled)
pred_df = df.predict(X_train_scaled)


import numpy as np
pred_stack = np.array([pred_lr,pred_kn,pred_df])

print(pred_stack)


print(pred_stack.shape) # --> 행과 열이 뒤바뀌었다.

pred_stack = pred_stack.T

print(pred_stack.shape)

from sklearn.ensemble import RandomForestClassifier

final_model  = RandomForestClassifier(n_estimators=100, max_depth=None, max_samples=0.5, max_features=0.3, random_state=1).fit(pred_stack, y_train)


v_score = final_model.score(pred_stack, y_train)
print(f'학습 final_model_RF : {v_score}')


pred_lr = lr.predict(X_test_scaled)
pred_kn = kn.predict(X_test_scaled)
pred_df = df.predict(X_test_scaled)


pred_stack = np.array([pred_lr,pred_kn,pred_df]).T

v_score = final_model.score(pred_stack, y_test)
print(f'테스트 final_model_RF : {v_score}')


