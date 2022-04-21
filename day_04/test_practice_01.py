# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 22:50:29 2022

@author: 오정민
"""
# 회귀 분석

import pandas as pd
# load : 연습을 위한 간단한 데이터 셋
# fetch : 실 데이터 셋 [ 상대적으로 데이터 갯수가 많음]
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)


# 설명변수의 EDA
print(X.info())

# 결측데이터 개수 확인

print(X.isnull()) # 빠진 데이터가 있는지 없는지 확인 
print(X.isnull().sum())

pd.options.display.max_columns = 100
print(X.describe())

# X 데이터를 구성하고 있는 각 특성들의 스케일 범위를 확인
# population 에서 스케일 차이가 있음 3 ~ 35682.000000까지

# 스케일 동일한 범위로 수정하기 위한 전처리
# 정규화 [StandardScaler] / 일반화 [MinMaxScaler]

# 종속변수
# 연속된 수치형 데이터 -> 숫자[수치]가 나열이 되어 있다.
print(y.head())
print(y.tail())

# 회귀분석은 중복이되는 경우가 흔치 않음 --> 분류 / 분석과 같이 value_counts 메소드를 사용하여 값의 개수 확인 과정 생략
# y 데이터 내부의 값의 분포 비율은 유지할 필요가 없음 
# print(y.value_counts())
# print(y.value_counts()/ len(y))

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y,
                                                 test_size=0.3,                                                 
                                                 random_state=1)

print(X_train.shape, X_test.shape)
print(len(y_train),len(y_test))


# 선형 방정식을 기반으로 회귀 예측을 수행할 수 있는 클래스
# 각 컬럼별[특성 , 피처] 최적화된 가중치와 절편의 값을 계산하는 과정을 수행
from sklearn.linear_model import LinearRegression

model = LinearRegression(n_jobs=-1)

# 학습
model.fit(X_train, y_train)

# 평가 (score 메소드)
# - 분류를 위한 클래스 : 정확도 : 전체 데이터 중 정답으로 맞춘 비율
# - 회귀를 위한 클래스 : R2 Score(결정계수) : ~1 까지의 범위를 가지는 평가값

# R2(결정계수) 계산 공식
# 1 - ((실제 정답과 모델이 예측한 값의 차이의 제곱 값 합계) / 
#     (실제 정답과 접답의 평균 값 차이의 제곱 값 합계))


# R2(결정계수)의 값이 0인 경우 :
# 머신러닝 모델이 예측한 값이 전체 정답의 평균으로만 예측하는 경우
# -> 머신러닝 모델이 학습이 부족하다

# R2(결정계수)의 값이 1인 경우 :
# 머신러닝 모델이 예칙흔 값이 실제 정답과 완벽하게 일치하는 경우
# -> 머신러닝 모델이 학습이 너무 잘되었다.(과대적합)

# R2(결정계수)의 값이 0보다 작은 경우 :
# 머신러닝 모델이 예측하는 값이 정답들의 평균조차 예측하지 못하는 경우
# -> 머신러닝 모델의 학습이 부족함(과소적합)

score = model.score(X_train, y_train)
print(f'Train : {score}')

# 테스트

score = model.score(X_test, y_test)
print(f'Test : {score}')

# 예측
X_test.iloc[:1] #앞에있는것은 행의 값을 제어 뒤에 것은 열의 값을 제어
# 테스트 데이터의 가장 앞 데이터를 사용하여 예측을 수행
pred = model.predict(X_test.iloc[:1])
print(pred)

# 머신러닝 모델이 학습 기울기[가중치] 절편을 확인
# 기울기[가중치]
print(model.coef_)

# 절편
print(model.intercept_)

pred = 3.25 * model.coef_[0] + 39.0 * model.coef_[1] + \
    4.503205 * model.coef_[2] + 1.073718 * model.coef_[3] + \
        1109.0 * model.coef_[4] + 1.777244 * model.coef_[5] + \
            34.06 * model.coef_[6] + -118.36 * model.coef_[7] + \
                model.intercept_
print(pred)

# model.coef_[0] = 4.41037995e-01


# R2(결정계수) : 데이터에 관계없이 동일한 결과의 범위를 사용하여 모델을 평가
# 평균절대오차 : 실제정답과 모델이 예측한 값의 차이를 절댓값으로 평균
# 평균절대오차비율 : 실제 정답과 모델이 예측한 값의 비율 차이를 절대값으로 평균값
# 평균제곱오차  실제 정답과 모델이 예측한 값의 차이의 제곱값 평균 [제곱하게 되면 값이 증폭되어서 값을 더 크게 할 수 있다.]

from sklearn.metrics import r2_score

# 평균절대오차
from sklearn.metrics import mean_absolute_error

# 평균절대오차비율
from sklearn.metrics import mean_absolute_percentage_error

# 평균제곱오차
from sklearn.metrics import mean_squared_error

# 평가를 위해서는 머신러닝 모델이 예측한 값이 필요 [ 공통 ]

pred = model.predict(X_train)

# 평균절대오차
mae = mean_absolute_error(y_train, pred)

print(f'MAE :  {mae}')


map = mean_absolute_percentage_error(y_train, pred)

print(f'MAP:  {map}')


# LinearRegression 클래스는 학습 데이터를 예측하기 위해서
# 각각의 특성 별로 최적화된 가중치의 값을 계산하는 머신러닝 알고리즘.
# - LinearRegression이 학습한 가중치는 학습 데이터에 베스트 핏이 되는 가중치

# - L1 제약 (Lasso) - 정말로 필요하다고[유의미한 것] 생각한것만 가중치를 주는 것
# - L2 제약 (Ridge) - 높은 가중치를 가질 수 없도록 만들어 주는 것 




























