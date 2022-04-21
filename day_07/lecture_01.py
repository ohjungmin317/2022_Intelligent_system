# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 15:02:45 2022

@author: 오정민
"""

import pandas as pd

pd.options.display.max_columns = 100
pd.options.display.max_rows = 10


fname_input ='./titanic.csv'

data = pd.read_csv(fname_input, header='infer', sep=',')


# 기존 데이터와 다르게 일부 문자열이 포함이 될 수 있다.
print(data.head())


print(data.info())
# dtypes: float64(2), int64(5), object(5) -> 데이터의 정보

print(data.describe(include=object))

# 수치형 데이터에서도 ID 성격의 데이터가 존재할 수 있음
# 데이터의 개수가 모두 1
# 하단의 length 정보가 전체 데이터의 개수와 동일함
print(data.PassengerId.value_counts())

# 문자열은 따로 표시가 안되서 float(2) + int(5) = 7개만 표시
# 결측데이터의 개수가 아닌 개수 + unique[중복되지 않는 것] + 가장많이 나온것 + 가장많이 나온것이 몇번 나왔는지 => 문자열 
# 각 컬럼의 중복을 제거시킨 데이터의 개수 
# cabin의 개수가 204개 기존 개수보다 너무 적음 => 결측데이터 [ 사용하기 힘들다 ]
# 머신러닝에 불필요한 정보를 확인한 후 제거할 수 있어야함 ['PassengerId', 'Name', 'Ticket', 'Cabin']




# 불필요한 컬럼의 제거 코드
# DataFrame의 drop 메소드
# - drop(columns=[제거할 컬럼명])
# inplace 매개변수 : 실행의 결과를 현재 변수에 반영할지 여부를 지정 

data2 = data.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=False)

print(data2.info()) # 시험에서 drop 한 이유를 물어보는 것이 나옴

# 결측데이터를 볼 수 있는 방법
print(data2.isnull().sum())


# 결측 데이터의 처리 방법
# 1, 결측 데이터가 포함된 레코드(열)을 제거 - Age Embarked의 데이터 개수가 많아 제거는 어려움
# 2, 결측 데이터가 포함된 레코드(행)을 제거 - Embarked 커럼의 경우 결측이 2개뿐이어서 행 자체를 제거 할 수 없음 
# 3, 기초 통계로 결측지를 대체한다. - (평균, 중심값, 최빈값 )
#                                   (최빈값 - 문자열)       

# 4, 지도학습 기반의 머신러닝 모델을 구축하여 예측한 값으로 결측데이터를 대치함
# (Age 컬럼 - 결측치가 많지만 결측이 아닌 데이터 개수가 월등히 많으므로 학습 고려대상)

# 5, 준지도학습 / 비지도학습 기반의 머신러닝
# 모델을 구축하여 모델의 예측값으로 결측치를 대체함 - 클러스터링
# (Cabin - 컬럼인 경우)

# dropna -> 결측치의 값을 제거 하겠다.
data3 = data2.dropna(subset=['Age', 'Embarked'])

print(data3.info())
print(data3.isnull().sum())

# X와 y를 나눌때 object를 먼저 확인해줘야한다.
# 문자열 데이터 전처리의 필요성
# 머신러닝 알고리즘이 문자열 처리 x -> 문자열 수치형 데이터로 변환

# 문자열 데이터의 전처리 위치
# 기본적으로 데이터의 분할전에 하는 것이 원칙


# 아래의 성별 데이터는 학습에도 남성 / 여성  | 테스트에도 남성 / 여성 포함 될 수 있을까?
print(data.Sex.value_counts())


# 아래의 성별 데이터는 학습에도 S/C/Q  | 테스트에도 S/C/Q 포함 될 수 있을까?
print(data.Embarked.value_counts())

# 문자열 데이터의 전처리는 기본적으로 매칭 방법을 사용
# 전처리 대상에 존재하는 문자열 데이터를 수치데이터로 연계하여 변환하는 방식을 사용

# 문자열 데이터의 전처리 방식
# 1, 라벨인코딩
# 특정 문자열 데이터를 정수와 매칭하여 단순 변환하는 방식
# 학습 데이터에 남성 / 여성이 존재하는 경우 
# EX ) 남성 / 여성 -> 0 / 1
# EX ) S / Q / C -> 0 / 1 / 2
# -> 정수와 매칭에서 숫자를 부여해주는 것
# 라벨인코딩은 일반적으로 정답 데이터(y)가 문자열로 구성되는 경우 사용을 할 수 있다.
# [단! 필수는 아니다.]

# 반면 라벨 인코딩은 설명변수 X에는 잘 사용안한다.
# ex ) Sex * W / Embarked * W

# 2, 원핫인코딩
# 유일한 문자열의 개수만큼 컬럼을 생성하여 그 중 하나의 위치에만 1을 대입하는 방식
# 1로 대입해주는 자리를 할당해주는 것 [ 대 / 소 관계 x + 독립적인 공간 형성 ]
# 학습 데이터 남성 / 여성이 존재하는 경우
# EX) 남성 / 여성 -> 1 0 / 0 1
# EX) S Q C -> 1 0 0 / 0 1 0 / 0 0 1
# 일반적으로 설명변수(X)에 포함된 문자열 데이터는 원핫인코딩을 많이 사용 
# 메모리의 낭비가 심해 유일한 값의 케이스를 줄일 수 있는 방향으로 접근하는 것이 유의미하다.


# 문자열 전처리를 위해 수치형 자료의 전처리와 과정이 독립적이므로 데이터를 분할하는 것이 작성에 편리함.

X = data3.iloc[:, 1:]
y = data3.Survived


# 수치형 데이터의 컬럼명
X_num = [cname for cname in X.columns if X[cname].dtype in ['int64','float64']]
# -> X.columns의 명이 다 들어가 있다.
# 문자형 데이터의 컬럼명
X_obj = X_num = [cname for cname in X.columns if X[cname].dtype not in ['int64','float64']]
# not in -> 그 외 것들 조건에 들어가지 않는것

print(X['Age'].dtype)
print(X['Sex'].dtype)
print(X['Pclass'].dtype)
# -> pandas의 숫자는 float와 int밖에 없다


X_num = X[X_num]
X_obj = X[X_obj]

print(X_num.info())
print(X_obj.info())


from sklearn.preprocessing import OneHotEncoder

# 원핫인코더 주요 파라메터
# - sparse : 희소행렬 생성 여부
# EX ) S / Q / C --> 100 / 010 / 001
#       희소행렬 --> (0) 1 / (1) 1 / (2) 1
# - handle_unknown : 학습과정에서 인지하지 못한 문자열 값에 대한 처리 프로세스 정의
# ('error' , 'ignore')
encoder = OneHotEncoder(sparse = False, handle_unknown='ignore')

# 사이킷 런의 전처리 클래스들은 학습용 클래스와 유사한 것이 아래와 같은 메소드를 제공
# 1, fit 메소드 : 전처리 과정에 필요한 정보를 수집
# 2, transform 메소드 : fit 메소드에서 인지한 결과를 바탕으로 데이터를 변환하여 반환하는 메소드 
# 3, fit_transform 메소드 : 1번과 2번의 과정을 한번에 

X_obj_encoded = encoder.fit_transform(X_obj)
print(X_obj.head())
X_obj_encoded
# Sex Embarked
# 0    male        S
# 1  female        C
# 2  female        S
# 3  female        S
# 4    male        S

print(X_obj_encoded[:5])
# [[0. 1. 0. 0. 1.]
#  [1. 0. 1. 0. 0.]
#  [1. 0. 0. 0. 1.]
#  [1. 0. 0. 0. 1.]
#  [0. 1. 0. 0. 1.]]

# 주의사항
# 모든 사이킷런의 전처리 클래스들은 transform 메소드의 결과가 numpy 배열로 반환
# pandas 데이터프레임이 아님

print(encoder.categories_)
print(encoder.feature_names_in_)

X_obj_encoded = pd.DataFrame(X_obj_encoded, columns=['s_f','s_m','e_C','e_Q','e_S'])


print(X_obj_encoded)


# 전처리된 데이터를 결합하여
# 설명변수 X를 생성


print(X_num.info())
print(X_obj_encoded.info())



X_num.reset_index(inplace=True)
X_obj_encoded.reset_index(inplace=True)


# concat 메소드 사용하여 데이터프레임을 결합

X = pd.concat([X_num, X_obj_encoded], axis =1 )


print(X.info())


# 종속변수 확인
# - 0 / 1에 대한 편향이 일부 존재
# 학습시 class 별 가중치를 제어할 필요성이 있음

print(y.value_counts())
print(y.value_counts() / len(y))


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,test_size=0.3,stratify=y)



from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
# 분류하는것이기 떄문에 classifier 랜덤이랑 그래디언트 차이는 -> 취합할것인가

model = RandomForestClassifier(n_estimators=100, max_depth=None, max_samples=1.0, class_weight='balanced', n_jobs = -1, random_state=0)

model.fit(X_train, y_train)

score = model.score(X_train, y_train)
print(f'Score(Train) : {score}')
# score는 1이 나온다 randomforest max_depth = None

score = model.score(X_test, y_test)
print(f'Score(Test) : {score}')


# 시험 2시간 필답형 [사이트 x] - 40분 (단답형 / 서술형)
# 작업형 1문제 절대적인 코드 / 니머지는 코드 고치거나 빈칸에 넣기







































