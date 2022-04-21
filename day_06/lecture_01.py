# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 15:06:33 2022

@author: 오정민
"""


# 앙상블 (Ensemble)
# - 다수개의 머신러닝 알고리즘 결합하여 각 모델이 예측한 결과를 취합 / 부스팅 방법을 통해 예측을 수행하는 방법 / 방법론

# 앙상블의 구현 방식 
# - (1) 취합 
# -> 앙상블을 구성하는 내부의 각 모델이 서로 독립적으로 동작
# -> 각각의 모델이 예측한 결과값에 대해서 다수결 방식을 수행(분류)
# -> 각각의 모델이 예측한 결과값에 대해서 평균의 취함(회귀분석)
# -> 내부의 각 모델은 서로 연관성이 존재하지 않음
# -> 취합 방식의 앙상블 앙상블 모델을 구축하는 경우 내부의 각 모델은 적절한 수준으로 과적합을 수행할 필요가 있음
# -> 학습 / 예측 수행 속도가 빠름
# -> (각 모델이 독립적으로 병렬 처리가 가능함) ((참고)
# - Voting / Baggin / RandomForest


# - (2) 부스팅
# -> 앙상블을 구성하는 내부의 각 모델들이 선형이 연결되어 학습 및 예측을 수행하는 방식
# -> 내부의 각가의 모델들은 다음의 모델에 영향을 주는 방식
# -> 내부의 각각의 모델들은 다음 모델에 영향을 주는 방식
# -> 부스팅 방식의 앙상블 모델은 내부의 각 모델에게서 강한 제약을 설정하여 점진적인 서능 향상을 도모함.
# -> (각 모델은 선형으로 연결되어 앞으로 모델들이 선형으로 연결되어 앞의 모델이 학습이 종료된 후 이유에도 학습이 됨)
# - AdaBoosting / GredientBoosting / XGBoost / LightGBM (참고)


import pandas as pd
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

y.head()
y.tail()
y.value_counts()


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)


# 앙상블 클래스 로딩
from sklearn.ensemble import VotingClassifier

# VotingClassifier parameters : 
# (1) estimator -> 내가 사용할 모델을 넣어줘야 한다.
# (2) voting -> hard : 단순히 개수만 해주는 것 (0과 1이 몇개 인지), soft : 0과 1의 비율을 가지고 예측을 해주는 것 
# -> 일반적으로 hard voting을 많이 해준다.
# (3) n_jobs -> 취합방식 [ -1 : 컴퓨터에서 모든 core를 사용해서 취합하는 함]

# 앙상블을 구현하기 위한 내부 모델의 클래스 로딩

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


#  앙상블을 구현할 3개의 모델 구현 [ 다수결 이기에 홀수로 만들어주는 것이 좋다.]
m1 = KNeighborsClassifier(n_jobs=-1)
m2 = LogisticRegression(n_jobs=1, random_state=1)
m3 = DecisionTreeClassifier(max_depth=3, random_state=1)

estimators = [('knn',m1),('lr', m2), ('dt',m3)]


model = VotingClassifier(estimators=estimators, voting='hard',n_jobs=1)

model.fit(X_train, y_train)

score = model.score(X_train, y_train)
print(f'score (Train) : {score}')

score = model.score(X_test, y_test)
print(f'score (Test) : {score}')

# 앙상블 모델의 결과
pred = model.predict(X_test[50:51])

print(f'Predict : {pred}')


# 앙상블 내부의 구성 모델 확인
print(model.estimators_[0])
print(model.estimators_[1])
print(model.estimators_[2])


# 앙상블 내부의 각 모델의 예측 값 확인 
pred = model.estimators_[0].predict(X_test[50:51])
print(f'Predict(knn) : {pred}')

pred = model.estimators_[1].predict(X_test[50:51])
print(f'Predict(lr) : {pred}')

pred = model.estimators_[2].predict(X_test[50:51])
print(f'Predict(dt) : {pred}')

