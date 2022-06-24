# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 18:00:44 2022

@author: 오정민
"""
import pandas as pd
pd.options.display.max_rows=100
pd.options.display.max_columns=100

# 1. 데이터 적재
fname_input = './test1.csv'
data = pd.read_csv(fname_input)

# 2. EDA 수행
print(data.head())

# - 데이터의 전체 개수 : 1460
# - 데이터 타입이 다양함 : int, float, object
# - 결측 데이터가 존재함

print(data.info())

# 데이터 내에서 아이디 컬럼과 같이 고유한 값을 가지는 컬럼은 제외함
print(data['Id'].value_counts())
data = data.iloc[:, 1:]
print(data.info())

# 결측 데이터의 개수 확인
print(data.isnull().sum())

# 결측 데이터의 개수를 확인하는 시리즈 변수를 생성
# 1차원 데이터를 인덱스와 함께 저장하는 Series 클래스 타입을 정리 
not_nan_series = data.isnull().sum()

# 타입 확인
print(type(not_nan_series))

# pandas series 타입을 사용하여 조건식 적용하는 예제
print(not_nan_series == 0)

# series 내부의 값을 조건식을 기준으로 추출하는 방법
# 결측 데이터가 존재하지 않는 컬럼 데이터만 추출
not_nan_series = not_nan_series[not_nan_series==0]
print(not_nan_series)

# series 타입의 인덱스 정보만 추출하여 결측 데이터가 존재하지 않는 컬럼명 정보를 추출
not_nan_series = not_nan_series.index.tolist() 
print(not_nan_series)

data = data[not_nan_series]
print(data.info())

# 데이터 전처리 1

obj_columns = [cname for cname in data.columns if data[cname].dtype == 'object']
num_columns = [cname for cname in data.columns if data[cname].dtype in ['int64','float64']]

# 라벨 인코더 import
from sklearn.preprocessing import LabelEncoder

# 문자열 인코딩 시, 주의 사항
# - 전처리 클래스이므로 fit 메소드를 사용해야함!
# - 라벨인코더의 경우 fit 메소드의 입력 값은 반드시 1차원만 가능
#   (라벨인코더는 각각의 컬럼별로 생성해야함!)
# - 원핫인코더의 경우 fit 메소드의 입력 값은 반드시 2차원만 가능
#   (원핫인코더는 다수개의 문자열 컬럼을 한번에 전처리할 수 있음)

print(data[obj_columns].head())

# 각 문자열 컬럼에 대한 인코더를 저장하기 위해서 딕셔너리 생성
dict_encoder = {}

# 문자열 컬럼의 개수만큼 반복을 수행하며 인코더를 생성 및 학습 
for cname in obj_columns : 
    encoder = LabelEncoder()
    encoder.fit(data[cname])
    dict_encoder[cname] = encoder
    
    # 원본 데이터를 라벨 인코딩 결과로 대체      
    data[cname] = encoder.transform(data[cname])

print(data[obj_columns].head())

print(data.info())


X = data.iloc[:, :-1]
print(X.info())

y = data.SalePrice


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=25)

print(X_train.shape, X_test.shape)


# 전처리 2
# 수치데이터에 대한 스케일링 처리
# RobustScaler 클래스는 이상치가 존재하는 데이터에 적합한 스케일링 처리 로직을 제공

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

# 종속변수가 포함되어 마지막 종속변수의 컬럼명 제거 -> y 값 제거
num_columns = num_columns[:-1]

# 스케일링 위한 전처리 클래스 (MinMax, Standard, Roubst) 들은 다수개의 컬럼에 대해서 1개의 스케일러가 처리 할 수 있음

scaler.fit(X_train[num_columns])

temp = scaler.transform(X_train[num_columns])


print(X_train[num_columns].head())
print(temp[:5])


# 학습 데이터에 스케일 처리된 결과를 대입
X_train[num_columns] = temp
print(X_train[num_columns].head(10))


# 테스트 데이터에 스케일 처리된 결과를 대입
print(X_test[num_columns].head())
X_test[num_columns] = scaler.transform(X_test[num_columns])
print(X_test[num_columns].head())

from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

model_knn = KNeighborsRegressor(n_neighbors=5, n_jobs=1).fit(X_train, y_train)
model_rf = RandomForestRegressor(n_estimators=100, max_depth=None, n_jobs=1).fit(X_train,y_train)
model_gb = GradientBoostingRegressor(n_estimators=100, max_depth=1, subsample=0.3, random_state=11).fit(X_train, y_train)


from sklearn.metrics import mean_absolute_error

score_knn = mean_absolute_error(y_test, model_knn.predict(X_test))
score_rf = mean_absolute_error(y_test, model_rf.predict(X_test))
score_gb = mean_absolute_error(y_test, model_gb.predict(X_test))

print(f'score_knn : {score_knn}')
print(f'score_rf : {score_rf}')
print(f'score_gb : {score_gb}')

# 모델 : randomforest 모델 사용
# 선정이유 : 현재 모델에서 평가 결과를 기반으로 가장 오차가 적은 randomforest 모델을 선정함


best_model = model_rf

score_r2 = best_model.score(X_test, y_test)
score_mae = mean_absolute_error(y_test, best_model.predict(X_test))

print(f'score_r2 : {score_r2}')
print(f'score_mae : {score_mae}')

# 현재 모델링 한 머신러닝의 모델 결과 분석은 상당히 우수함 
# 판단 근거로는 결정계수의 값이 0.88로 대다수의 테스트 데이터에 대해서 근접한 값으로 예측하고 있는것으로 확인이 된다.


