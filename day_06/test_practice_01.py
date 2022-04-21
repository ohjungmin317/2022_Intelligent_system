# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 14:04:47 2022

@author: 오정민
"""

# 앙상블 
# 다수개의 모델을 결합 하나의 예측을 할 수 있는 결합 모델
# 다수결의 원칙이 적용 [ 분류 ]
# 평균 원칙이 적용 [ 회귀 ]

# 앙상블은 가장 좋은 평가 성적을 반환하지 않음
# - 앙상블은 하나의 모델 객체를 사용하지 않고 다수개의 모델 결과를 사용하여 다수결/평균을 취하므로
#   앙상블을 구성하는 많은 모델에서 가장 좋은 성적의 모델보다는 항상 평가가 떨어질 수 있음
#   일반화 성능을 극대화하는 모델 [ 테스트 데이터에 대한 성능 ]

# 앙상블을 구현하는 방법
# - 취합 / 부스팅 -
# 취합 : 앙상블을 구성하고 있는 각각의 모델이 독립적으로 동작
# 각각의 모델이 독립적으로 학습하고 예측한 결과를 반환하여 최종적으로 취합된 결과를 다수결 / 평균으로 예측
# -> EX) voting, Bagging, RandomForest
# 취합기반의 앙상블 내부의 모델들은 각각 일정 수준 이상의 예측성능을 달성
# 학습과 예측의 속도가 빠름 [ 병렬처리가 가능한 구조 ]


# 부스팅 : 앙상블을 구성하는 각각의 모델이 선형으로 결합되어 
#         점진적으로 학습의 성능을 향상시켜 나가는 방법 [ 순차적으로 나감 ]
# 부스팅의 첫번째 모델이 예측결과 * 가중치 + 두번째 모델이 예측한 결과 * 가중치 + n번째 모델이 예측한 결과 * 가중치 = 결과
# -> EX) AdaBoosting , GradientBoosting, XGBoost, LightGBM -> 시험문제 나옴
# 부스팅 기반의 앙상블 내부 모델은 강한 제약조건을 설정하여서 점진적으로 성능 향상이 될 수도 있도록
# 학습과 예측의 속도가 느리다 [ 순차적으로 처리해야 하기 때문에 ]


import pandas as pd
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns = data.feature_names)
y = pd.Series(data.target)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,stratify=y,random_state=1)


# 앙상블 기반의 클래스 로딩

# voting -> 여러개의 모델 중에서 가장 좋은 모델을 선택함 -> 다수결의 원칙

from sklearn.ensemble import VotingClassifier


# 앙상블을 구성하는 각 모델의 클래스 로딩
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

m1 = KNeighborsClassifier(n_jobs = 1)
m2 = LogisticRegression(random_state=1, n_jobs = 1)
m3 = DecisionTreeClassifier(max_depth=3, random_state=1)



estimators = [('knn',m1),('lr', m2),('dt', m3)]

model = VotingClassifier(estimators=estimators, voting = 'hard', n_jobs = 1)
#  estimators : 예측기 -> 모델이름 입력
#  voting : hard: 각 하나하나가 똑같은 비율을 가지고 있음  / soft : 예측한 값의 확률을 기반으로
#  n_jobs = -1 


model.fit(X_train, y_train)


score = model.score(X_train, y_train)
score


score = model.score(X_test, y_test)
score

pred = model.predict(X_test[:1])
pred

model.estimators_[0]
