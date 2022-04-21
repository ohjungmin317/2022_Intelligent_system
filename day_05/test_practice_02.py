# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 20:37:04 2022

@author: 오정민
"""

# 트리모델 [ 분류 ]

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


from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(max_depth=None, class_weight=None, random_state=5)

# criterion : 분류할때 기준으로 삼을 것 'gini , entropy'
# max_depth -> none : 끝까지
# min_samples_split : 최소 샘플의 갯수
# min_samples_leaf : leaf 노드 1개를 위해
# random_state : 상태를 고정

model.fit(X_train, y_train)

model.score(X_train, y_train)

model.score(X_test, y_test)
  

# 특성(컬럼) 중요도의 값
print(model.feature_importances_)
# 안에 중요도 값은 다 더하면 1이 나오게 된다.

































