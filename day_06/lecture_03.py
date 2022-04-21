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

# - 랜덤포레스트 : 배깅 방법론에 결정 트리를 조합하여 사용한 패턴이 빈번하게 발생하여
# 해당 구조를 하나의 앙상블 모형으로 구현해 놓은 클래스

# 랜덤포레스트 : 배깅 + 결정트리
from sklearn.ensemble import RandomForestClassifier


#학습이나 test 를 조절하기 위해서 sample 이나 estimators를 조절하는 것이 좋다.
model = RandomForestClassifier(n_estimators=100, 
                               max_depth=None,
                               n_jobs=-1,
                               max_features=0.5,
                               max_samples=0.3,
                               random_state=1)


model.fit(X_train, y_train)

score = model.score(X_train, y_train)
print(f'score (Train) : {score}')

score = model.score(X_test, y_test)
print(f'score (Test) : {score}')

