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


from sklearn.svm import SVC,LinearSVC

model = SVC(C=1.0,gamma='scale',class_weight='balanced',random_state=1)

# C -> 값이 커질수록 제약이 작아진다. / 값이 작아질수록 제약이 커진다.
# kernel : 'rbf'
# gamma : 밀집데이터를 상세하게 분류하는것

model.fit(X_train, y_train)

model.score(X_train, y_train)

model.score(X_test, y_test)
  

# svm 가중치의 값 확인 
# 커널 방법을 linear로 선택하면 가능 
print(model.coef_)

print(model.intercept_)

































