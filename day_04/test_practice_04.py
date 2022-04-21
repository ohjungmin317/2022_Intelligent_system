
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



# 성능 향상!

from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 3 ,include_bias=False).fit(X_train)

X_train = poly.transform(X_train)
X_test = poly.transform(X_test)

print(X_train.shape, X_test.shape) # 차수의 증가로 [속성, 특성 , 피처]의 값 증가

# 2, 스케일 처리
# - 정규화 : 데이터를 구성하는 각 컬럼의 값을 평균은 0 표준편차는 1로 스케일을 조정
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# 스케일 조정으로 값의 스케일만 변화가 적용이 되고 개수는 변하지 않음


# 선형 방정식을 기반으로 회귀 예측을 수행할 수 있는 클래스
# 각 컬럼별[특성 , 피처] 최적화된 가중치와 절편의 값을 계산하는 과정을 수행
from sklearn.linear_model import LinearRegression

# 선형 모델에 제약조건 (L2, L1 제약조건)을 추가한 클래스

# 선형 모델에서 L2 제약 조건을 추가한 Ridge 클래스
# L2 제약 조건 : 모든 특성에 대한 가중치 값을 0 주변으로 위차하도록 제어하는 제약 조건
# 학습을 하기 때문에 테스트 데이터에 대한 일반화 성능이 감소된다.
# 이러한 경우 모든 특성 데이터를 적절히 활용할 수 있도록 L2 제약 조건을 사용할 수 있으며 L2 제약 조건으로 인하여
# 모델의 일반화 성능이 증가하게 된다.

# 선형 모델에서 L1 제약 조건을 추가한 Lasso 클래스
# L1 제약조건 : 모든 특성 데이터 중 특정 특성에 대해서만 가중치의 값을 할당하는 제약조건
# [ 대다수의 가중치의 값을 0으로 제약]
# L1 의 제약조건은 특성 데이터가 만흥ㄴ 데이터를 학습하는 경우
# 빠르게 학습을 할 수 있는 장점을 가짐 모든 특성 데이터 중 중요도가 높은 특성을 구분할 수 있음


# Ridge Lasso 클래스의 하이퍼 파라메타 alpha
# alpha의 값이 커질 수록 제약을 크게 설정
# (alpha의 값이 커질수록 모든 특성들의 가중치의 값은 0 주변으로 위치한다.)
# alpha의 값이 작아질 수록 제약을 약해짐
# (alpha의 값이 작아질 수록 모든 특성들의 가중치의 값은 0에서 멀어진다.)
# alpha의 값이 작아질수록 LinearRegression 클래스와 동일해짐 

from sklearn.linear_model import Ridge, Lasso


# 머신러닝 객체 생성
lr_model = LinearRegression(n_jobs=-1)
ridge_model = Ridge(alpha=1, random_state=1) # alpha 값은 가중치를 주는 것
lasso_model = Lasso(alpha=0.01, max_iter=100000, random_state=1)

# 학습
lr_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)

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

score = lr_model.score(X_train, y_train)
print(f'Train(LR) : {score}')

score = ridge_model.score(X_train, y_train)
print(f'Train(Ridge) : {score}')


score = lasso_model.score(X_train, y_train)
print(f'Train(lasso) : {score}')

# Lasso 가 0 이 나온다 : Train(lasso) : 0.0 -> 너무 적합을 하지 못함

# -> 더이상 차원을 확장해도 정확도가 높아지지 않는다.

# Lasso 클래스를 사용하여 모델을 구축하게 되면 대다수의 특성 가중치는 0으로 수렴 
print(lasso_model.coef_)

# Lasso alpha = 1.0 -> 이것도 제약이 커서 가중치의 값이 0이 나온다.
# [ 1.47014609e-01  5.94733939e-03  0.00000000e+00 -0.00000000e+00
 # -8.56831227e-06 -0.00000000e+00 -0.00000000e+00 -0.00000000e+00]

# Lasso alpha = 0.0001 -> alpha의 값을 작게하여 제약을 작게함
# [ 4.40682770e-01  9.69506881e-03 -1.04103865e-01  6.18582657e-01
#  -5.58400616e-06 -3.28657714e-03 -4.23056089e-01 -4.37717432e-01]


# Train(lasso) : 0.6097138442910215 -> 제약을 작게해서 LR의 비율과 비슷해진다.


# 테스트

score = lr_model.score(X_test, y_test)
print(f'Test : {score}')


score = ridge_model.score(X_test, y_test)
print(f'Test : {score}')


score = lasso_model.score(X_test, y_test)
print(f'Test : {score}')

