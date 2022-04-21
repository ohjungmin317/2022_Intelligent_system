# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 20:58:43 2022

@author: 오정민
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

print(data.keys())

X = pd.DataFrame(data.data, columns = data.feature_names)

y = pd.Series(data.target)


# X(설명변수)
print(X.info())

pd.options.display.max_columns = 30 # 30개 모두 display 해주는 것 

# 데이터를 구성하는 각 컬럼들에 대해 기초 통계 확인 
# 데이터의 개수
# 평균 / 표준편차
# 최소 / 최대값
# 4분위수

# 스케일 : 특정 컬럼 값의 범위
# - 각 컬럼 별 스케일의 오차가 존재하는 경우 머신러닝 알고리즘의 종류에 따라 스케일 전처리 필요
print(X.describe()) 

# y(종속변수)
print(y)

# 종속변수의 범주형인 경우 
# 범주형 값의 확인 및 개수 체크 
print(y.value_counts()) # y의 값에서 중복인 경우 중복을 제거하고 개수를 보여주는 것


# 범주형 종속변수의 경우 값의 개수 비율이 중요 
print(y.value_counts() / len(y)) # 비율 계산


# 데이터 전처리는 학습 데이터에 대해서 수행
# 테스트 데이터는 학습 데이터에 반영된 결과를 수행

# 스케일처리가 완벽하게 0~1로 나뉘면 변수가 생기는 경우 예측이 안됨.
# 전처리가 완벽하게 잘 끝나면 학습 성능은 올라가지만 실전에서는 결과가 좋지 않음 

# * 데이터의 분할 * 
# 학습데이터(70/80(%)), 테스트데이터(30/20(%)) [ 머신러닝 케이스 ]

from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    test_size=0.2,
                                                    stratify=y,
                                                    random_state=30)


# 사용 예제 -> X_train / X_test / y_train / y_test = train_test_split(X, y,
#                                                         test_size = 테스트데이터비율
#                                                         stratify = 범주형데이터인경우 [y]를 입력
#                                                         random_state = 임의의 정수값 )
print(len(X_train), len(X_test))

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

# 전처리과정의 데이터 학습은 학습데이터를 기준으로 수행
# 학습데이터의 최소 최대값을 기준으로 스케일링 수행 준비

# 학습데이터 스케일 처리 수행
scaler.fit(X_train)

X_train = scaler.transform(X_train)

# 테스트 데이터는 학습데이터 기준으로 변환
X_test = scaler.transform(X_test)




# train_test_split 함수의 random_state는 데이터 분할된 값이 항상 동일하도록 유지하는 역할
# 머신러닝의 랜덤값은 고정시키고 머신러닝의 학습 방법만 제어함.
print(y_train[:10])

# stratify : 데이터가 분류형 데이터 셋인 경우에만 사용 (y 데이터가 범주형인 경우에)
# random_state의 값에 관계없이 비율이 유지가 된다.

print(y_train.value_counts()/ len(y_train))

# * 데이터 전처리 *
# 스케일 처리 [MinMax, Standard Robust]


# * 머신러닝 모델의 구축 *
from sklearn.neighbors import KNeighborsClassifier

# 머신러닝 모델 객체 생성
# 각 머신러닝 알고리즘에 해당하는 하이퍼 파라메터의 제어가 필수

model = KNeighborsClassifier(n_neighbors=11, n_jobs=-1)

# n_jobs = -1 은 cpu의 모든 core를 사용하여 가동
# 머신러닝 모델 객체 학습
# fit 메소드 사용 

# X는 무조건 2차원 데이터 셋 -> DataFrame
# y는 무조건 1차원 데이터 셋 -> Series

model.fit(X_train, y_train)

# 학습이 완료된 머신러닝 모델 객체 평가
# score 메소드를 사용

# 정확도 : 전체 데이터에서 정답인 데이터 비율
# 머신러닝 클래스의 타입이 회귀형인 경우 score 메소드는 결정계수를 반환
score = model.score(X_train, y_train)
print(f'Train = {score}')

score = model.score(X_test, y_test)
print(f'Test = {score}')

# 학습된 머신러닝 모델을 사용하여 예측 수행
# - predict 메소드 사용
# 예측할 데이터는 X는 반드시 2차원으로 선언

pred = model.predict(X_train[:2])
print(pred)
print(y_train[:2])

# 학습된 머신러닝 모델이 분류형인 경우
# 확률 값으로 예측할 수 있음 [일부 클래스에서는 제공 x]
# predict_proba

proba = model.predict_proba(X_test)
print(proba) # 예측한것중에 1에 강하게 예측하는 것은 놔두고 아닌것은 버린다 [일정이상 확률로 예측하는 경우에만]

