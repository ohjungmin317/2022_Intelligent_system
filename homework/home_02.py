# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 17:06:05 2022

@author: 오정민
"""

# 당뇨 수치 데이터(load_diabetes)를 사용하여 앙상블 기반의 회귀분석 모델을 구축하세요.
# 모델을 구축한 후 학습, 테스트 데이터에 대한 평균 절대 오차를 출력하여
# 모델의 적합성을 평가하세요.
#  - 앙상블 모델은 배깅, 그레디언트 부스팅을 사용하세요.

import pandas as pd
from sklearn.datasets import load_diabetes

pd.options.display.max_columns = 100

data = load_diabetes()

X = pd.DataFrame(data.data, columns = data.feature_names)
y = pd.Series(data.target)

X.head()
X.info()
X.isnull().sum()

X.describe()

y.head()
y.value_counts()
y.value_counts()/len(y)



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1)


from sklearn.ensemble import GradientBoostingRegressor


from sklearn.metrics import mean_absolute_error


model = GradientBoostingRegressor(n_estimators=100, 
                          learning_rate=0.3, subsample=0.2, max_depth=1, random_state=1)

model.fit(X_train, y_train)

train_pred = model.predict(X_train)
test_pred =  model.predict(X_test)


score = model.score(X_train, y_train)
print(f'Score(Train) : {score}')

score = model.score(X_test, y_test)
print(f'Score(Test): {score}')

train_mae = mean_absolute_error(y_train, train_pred)

print(f'Train_Mae : {train_mae}')

test_mae = mean_absolute_error(y_test, test_pred)

print(f'Test_Mae : {test_mae}')



