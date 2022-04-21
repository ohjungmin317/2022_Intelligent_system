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
# 배경 : 특정 머신러닝 알고리즘을 기반으로 데이터의 무작위 추출을 사용하여 
#       각 모델들이 서로다른 데이터를 학습하는 방식으로 앙상블을 구현하는 방법
# -> 과적합을 최적화 시키기 위해 / 기본적으로 성능 이상은 나오기 때문에

from sklearn.ensemble import BaggingClassifier

# BaggingClassifier parameters : 
# (1) n_estimator -> 사용할 모델을 몇개 사용할지 정해준다. 
# (2) max_features
# (3) n_jobs -> 취합방식 [ -1 : 컴퓨터에서 모든 core를 사용해서 취합하는 함]

# 앙상블을 구현하기 위한 내부 모델의 클래스 로딩
from sklearn.tree import DecisionTreeClassifier


#  앙상블을 구현할 3개의 모델 구현 [ 다수결 이기에 홀수로 만들어주는 것이 좋다.]
base_estimator = DecisionTreeClassifier(random_state=1)

model = BaggingClassifier(base_estimator=base_estimator, n_estimators=50, 
                          max_samples=1.0,
                          max_features=0.3,
                          random_state=1)

model.fit(X_train, y_train)

score = model.score(X_train, y_train)
print(f'score (Train) : {score}')

score = model.score(X_test, y_test)
print(f'score (Test) : {score}')

