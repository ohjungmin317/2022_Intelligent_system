# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:17:09 2022

@author: 오정민
"""
# 트리모델 (분류)

import pandas as pd
pd.options.display.max_columns = 100
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)

y = pd.Series(data.target)

X.info()
X.isnull().sum()
X.describe(include='all') 
# 문자열 데이터를 추출할 수 있음 -> 스케일의 오차가 일부 존재하는 데이터

y.head()
y.value_counts() / len(y) # 중복을 제거해준 count series -> 비율로 봐주는 것이 좋다

from sklearn.model_selection import train_test_split

splits = train_test_split(X,y,test_size=0.3,random_state=10,stratify=y) # 4개에 받아야하는데 변수 하나에 받아도 상관 없다.

X_train = splits[0]
X_test = splits[1]
y_train = splits[2]
y_test =  splits[-1]

X_train.head()
X_test.head()

X_train.shape
X_test.shape

y_train.value_counts() / len(y_train)
y_test.value_counts() / len(y_test)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=None, class_weight='balanced', random_state=1)
# -parameters-
# gini / entropy : 분류할때 기준으로 잡는것 -> 둘중에 선택하는 것에 따라 값이 달라진다
# max_depth : tree의 최대 깊이를 제어해주는 것 -> 과대적합이 기본값 
# min_samples_split : 데이터가 노드 조건에 따라 분류 / default =2
# min_samples_leaf : default = 1
# random_state
# class_weight

# - attributes -
# feature_importances_


model.fit(X_train, y_train)
model.score(X_test, y_test)

# 특정(컬럼)의 중요도 값 확인
print(f'feature_importances_ : {model.feature_importrances_}')

# 절편의 값 확인 
print(f'intercept_ : {model.intercept_}')


proba =model.predict_proba(X_train[:5])
proba 

pred = model.predict(X_train[:5])
pred

df = model.decision_function(X_train[:-10])
df

pred = model.predict(X_train[:-10])
pred

y_train[:5]

# 분류모델의 평가 방법
# 1, 정확도
# ~ 전체 데이터에서 정답으로 맞춘 비율
# ~ 머신러닝 모델의 score 메소드 
# ~ 분류하고자 하는 각각의 클래스의 비율이 동일한 경우에만 사용 

# 2, 정밀도
# ~ 집합: 머신러닝 모델이 예측한 결과
# ~ 위의 집합에서 각각의 클래스 별 정답 비율
# ~ 아주 확실한것만 측정을 하기 때문에

# 3, 재현율
# ~ 집합 : 학습 데이터 셋
# ~ 위의 집합에서 머신러닝 모델이 예측한 정답 비율 

# 혼동행렬
from sklearn.metrics import confusion_matrix

pred = model.predict(X_train)
cm = confusion_matrix(y_train, pred)
cm

y_train.value_counts()

#[[142,  6],
 #[  8, 242]] # 클래스별 가중치를 조절해서 정밀도를 조절할 수 있다.

# 머신러닝 모델의 예측    0   1
# 실제 0인 데이터     [[141,  7]
# 실제 1인 데이터     [  5, 245]]
# 0번 클래스 141개 정답 / 1인 클래스 245개 정답

# 정밀도 (0) : 141 / (141 + 5) 
# 재헌율 (0) : 141 / (141  + 7)

# 정확도를 반환시켜주는 함수
from sklearn.metrics import accuracy_score
# 정밀도
from sklearn.metrics import precision_score
# 재현율
from sklearn.metrics import recall_score

pred = model.predict(X_train)
# 정밀도
ps = precision_score(y_train, pred, pos_label=0)

ps

# 재현율
rs = recall_score(y_train, pred, pos_label=0)

rs

















