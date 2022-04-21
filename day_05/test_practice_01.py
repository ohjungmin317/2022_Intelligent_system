# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 20:37:04 2022

@author: 오정민
"""

# 선형모델 [ 분류 ]

import pandas as pd

pd.options.display.max_columns = 100

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns = data.feature_names)
y = pd.Series(data.target)


X.info()

X.isnull().sum()

X.describe()

y.head()
y.value_counts()
y.value_counts() / len(y)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3, 
                                                         stratify=y, 
                                                     random_state=10)

X_train.head()
X_test.head()


y_train.value_counts() / len(y_train)
y_test.value_counts() / len(y_test)


from sklearn.linear_model import LogisticRegression

model = LogisticRegression(penalty='l2', C = 1.0, class_weight={0:1000, 1:1}, solver='lbfgs', max_iter=100, 
                           n_jobs=1, random_state=5, verbose=3)

# penalty : 제약 -> l1(Lasso) , l2(Ridge), elasticnet (search: l1 + l2)
# C : C의 값이 커지면 제약이 작아진다. / C의 값이 작아지면 제약이 커진다.
# class_weight = 'balanced' => 정답비율을 균형있게 해주는 것이다.  / 'None' => default
# solver = 'lbfgs'

# class_weight = 'balanced' -> 0과 1의 균형을 맞추는 것 

model.fit(X_train, y_train)

model.score(X_train, y_train)

model.score(X_test, y_test)
  

# 가중치 값 
print(model.coef_)

# 절편의 값

print(model.intercept_)


proba = model.predict_proba(X_train[:5])
proba


pred = model.predict_proba(X_train[:5])
pred

df = model.decision_function(X_train[-10:])
df


pred = model.predict(X_train[-10:])
pred


y_train[:5]


# 분류모델의 평가 방법

# 1, 정확도 : 전체 데이터에서 정답으로 맞춘 비율 
# 머신러닝 모델의 => score
# 분류하고자 하는 각각의 클래스의 비율이 동일한 경우에만 사용


# 2, 정밀도
# -집합 : 머신러닝 모델이 예측한 결과
# - 위의 집합에서 각각의 클래스 별 정답 비율
# 머신러닝이 예측한 모델이 = 정답인 비율


# 3, 재현율
# 집합 : 학습 데이터 셋
# 위의 집합에서 머신러닝 모델이 예측한 정답 비율
# 내가 가지고 있는 데이터 중에 상한가 치는 날이 100일인데 머신러닝은 1일 -> 조금 틀려도 거의다 맞춘다.


# 혼동행렬

from sklearn.metrics import confusion_matrix

pred = model.predict(X_train)
cm = confusion_matrix(y_train, pred)
cm

y_train.value_counts()

# ([[140,   8], -> 실제 0인 데이터
#        [  7, 243]] -> 실제 1인 데이터

# 0인 정답 -> 140  1인 정답 -> 243

# 정밀도 (0) : 140 / (140+7)
# 재헌율 (0) : 140 / (140+8)

# 정밀도 (1) : 243 / (243+8)
# 재헌율 (1) : 243 / (243+7)


# {0:1000 , 1: 1} => dictonary로 정해주는것
# 정답 : 0인 데이터를 다 맞춤 
# 정밀도가 훨씬 떨어진다.
# ([[148,   0],
#        [ 91, 159]]

# 정확도
from sklearn.metrics import accuracy_score
# 정밀도
from sklearn.metrics import precision_score
# 재현율
from sklearn.metrics import recall_score

pred = model.predict(X_train)


ps = precision_score(y_train, pred, pos_label=1)

ps

rs = recall_score(y_train, pred, pos_label=1)

rs


















































