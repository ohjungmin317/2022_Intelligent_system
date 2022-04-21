# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 16:03:06 2022

@author: 오정민
"""

import pandas as pd

pd.options.display.max_columns=100

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

X.head()

y.head()
y.tail()
y.value_counts()


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.3, stratify=y, random_state=1)


# 앙상블 클래스 로딩

# - 부스팅 계열 
# - 앙상블을 구현하는 내부의 각 모델들이 선형으로 연결되어 
#   학습 및 예측을 수행하는 방법론 [별도가 아닌 모두가 연결이 되어 있음]

# 1, AdaBoost : <데이터 중심>의 부스팅 방법론을 구현 [틀린 것에 중점(맞았나 틀렸나만 중점)] 
# -> <데이터 중심> [직전 모델이 잘못 예측한 데이터(오분류)에 가중치를 부여하는 방법]


# 2, GradientBoosting : 오차의 중심을 둔 부스팅 방법론을 구현  [얼마만큼 틀린 것에 중점(얼마만큼 틀렸나)-> 간격을 매꾸기 위해서]
# -> [각각의 학습 데이터에 대해서 오차의 크기가 큰 데이터에 가중치를 부여하여 전체 오차를 줄여가는 방식]

# 부스팅 계열의 데이터 예측 예시
# 1번쨰 모델의 예측 값 * 가중치 (1번째 모델의 가중치[약한가중치]) + 
# 2번쨰 모델의 예측 값 * 가중치 (2번째 모델의 가중치) + 
# ....
# N번쨰 모델의 예측 값 * 가중치 (N번째 모델의 가중치)


# GradientBoosting 클래스는 부스팅을 구현하기 위한 기본 모델이 결정트리로 고정되어있음
from sklearn.ensemble import GradientBoostingClassifier

# 앙상블 구현하기 위한 기본 클래스 로딩

model = GradientBoostingClassifier(n_estimators=50, 
                                   learning_rate=0.1, 
                                   max_depth=1,
                                   subsample=0.3,
                                   max_features=0.3,
                                   random_state=1,verbose=3)




model.fit(X_train, y_train)

score = model.score(X_train, y_train)
print(f'score (Train) : {score}')

score = model.score(X_test, y_test)
print(f'score (Test) : {score}')

