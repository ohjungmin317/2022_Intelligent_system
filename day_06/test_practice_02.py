# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 14:04:47 2022

@author: 오정민
"""
import pandas as pd
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns = data.feature_names)
y = pd.Series(data.target)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,stratify=y,random_state=1)


# 앙상블 기반의 클래스 로딩

# RandomForest : 배깅 앙상블에 결정트리를 조합한 모델이 주로 사용이 되어 하나의 클래스로 정의한 모델

from sklearn.ensemble import RandomForestClassifier


# 앙상블을 구성하는 각 모델의 클래스 로딩
from sklearn.tree import DecisionTreeClassifier


model = RandomForestClassifier(n_estimators=100, max_depth=None, max_samples=1.0, max_features=0.7, n_jobs=-1, random_state=1)
# randomforest 취합인 이유 -> n_jobs 

# randomforest로 decision로 회귀모델 해줄 수 없는것 => domain의 target 값 : y값
# -> y data의 한계값을 벗어날 수 없어서



model.fit(X_train, y_train)


score = model.score(X_train, y_train)
score


score = model.score(X_test, y_test)
score

pred = model.predict(X_test[:1])
pred

model.estimators_[0]

# 01:40:58