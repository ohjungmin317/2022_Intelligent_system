# -*- coding: utf-8 -*-
"""
Created on Sat Apr 16 19:31:47 2022

@author: 오정민
"""

import pandas as pd

pd.options.display.max_columns = 100
pd.options.display.max_rows = 10

fname_input = './titanic.csv'
data = pd.read_csv(fname_input,header='infer',sep=',')
# f9를 누르면 경로가 들어가서 + f5 누르면 파일이 들어가기때문에 인식이 된다.

print(data.head())
print(data.info())

# dtypes: float64(2), int64(5), object(5)
 # 0   PassengerId  891 non-null    int64  
 # 1   Survived     891 non-null    int64  
 # 2   Pclass       891 non-null    int64  
 # 3   Name         891 non-null    object 
 # 4   Sex          891 non-null    object 
 # 5   Age          714 non-null    float64
 # 6   SibSp        891 non-null    int64  
 # 7   Parch        891 non-null    int64  
 # 8   Ticket       891 non-null    object 
 # 9   Fare         891 non-null    float64
 # 10  Cabin        204 non-null    object 
 # 11  Embarked     889 non-null    object 
 
print(data.describe())

print(data.describe(include='object'))
# 고유한 값이 개수가 많은 것은 제외해주는 것이 좋다. -> ex) Name / Ticket
# count 에서 data의 개수가 유난히 작다 -> 결측이 많아서 빠진 데이터가 많다.
# 제거할 대상 -> 'Name' , 'Ticket' , 'Cabin'
# 수치형[int , float] data 에서도 고유값이 있을 수 있음

#                           Name   Sex  Ticket    Cabin Embarked
# count                       891   891     891      204      889
# unique                      891     2     681      147        3
# top     Braund, Mr. Owen Harris  male  347082  B96 B98        S
# freq                          1   577       7        4      644


print(data.PassengerId.value_counts())
# 1      1
# 599    1
# 588    1
# 589    1
# 590    1
#       ..
# 301    1
# 302    1
# 303    1
# 304    1
# 891    1
# -> 다 unique 하기 때문에 [ 너무 데이터가 없기 때문에 ]

data2 = data.drop(columns=['Name', 'Ticket', 'PassengerId', 'Cabin'])
# 시험문제 : 데이터들을 왜 지웠는가?

# inplace = Ture -> 현재 실행한 결과를 그 값에 반영을 할 것인가?

print(data2.info())
print(data2.isnull().sum()) # 결측데이터를 확인할 수 있음 

# 결측데이터 처리하는 방법
# 1, 결측 데이터가 포함된 레코드(열)을 제거
# -> age 와 embarked의 괜찮은 데이터가 많기 때문에 제거하기가 좀 그래 
# 2, 결측 데이터가 포함된 레코드(행)을 제거
# -> embarked 결측데이터가 2개밖에 없기 때문에 행을 제거해도 괜찮다.

# 3, 기초 통계로 결측지를 대체함 -> EX) 평균 중심값 최빈값 [ 수치형 ]으로 대체
#                             -> EX) 최빈값 [ 문자형 ]

# 4, 지도학습 기반의 머신러닝 모델을 구축하여 예측한 값으로 결측데이터를 대치함
# age -> 결측치는 많지만 정답인 데이터가 많기 때문에 학습 고려 대상

# 5, 준지도 비지도 학습 기반의 머신러닝 모델을 구축 -> 클러스터링
# [결측데이터 보다 정답인 데이터보다 더 많을 때 근데 무조건 위 데이터를 사용해야할 때]


# 결측치를 지워지는 것 -> 아예 데이터를 삭제하는 것과 다르다. 
data3 = data2.dropna(subset=['Age','Embarked'])

print(data3.info())
print(data3.isnull().sum())

# 문자열 데이터 전처리 필요성
# - 머신러닝 알고리즘은 문자열로 처리가 아예 가능하지 않음 
# 문자열 -> 수치형으로 변환해줘야 한다.

# 문자열 데이터 전처리 위치
# (1) 기본적으로 데이터 분할전에 하는 것이 원칙

# ex Sex data
print(data.Sex.value_counts())

# ex Embarked data
print(data.Embarked.value_counts())

# (2) 문자열 데이터의 전처리는 기본적으로 매칭 방법을 사용 

# (3) 문자열 데이터 전처리 방식

# 1. 라벨인코딩 
# -> 특정 문자열 데이터를 정수와 매칭하여 단순 변환하는 방식
# 남자 - 0 /  여자 - 1  |  S - 0 / Q - 1 / C - 2
# 라벨인코딩은 정답 데이터(y)에서 많이 사용 -> 설명변수(x) 에서 사용 
# why? 선형모델을 사용할 때 가중치의 값이 0일때에는 값의 의미가 사라진다.


# 2, 원핫인코딩
# -> 유일한 문자열의 개수만큼 컬럼을 생성하여 그 중 하나의 위치에만 1을 대입하는 방식 
# -> ex) 남성 / 여성 => 남성에 1이 들어오면 여성에는 0이 들어어고 반대로 여성에는 1이 들어오면 남성에는 0이 들어옴
# => 대소관계가 사라지게 된다.
# 메모리 낭비가 심해 유일한 값의 케이스를 줄일 수 있는 방법을 접급

X = data3.iloc[:, 1:]
y = data3.Survived

# 수치형 데이터의 컬럼명
X_num = [cname for cname in X.columns if X[cname].dtype in ['float64','int64'] ]
X_obj = [cname for cname in X.columns if X[cname].dtype not in ['float64','int64']]


print(X['Age'].dtype) # 해당 type이 수치형인지 문자열인지 확인하는 법 

X_num = X[X_num] # dataframe으로 변경해주는것
X_obj = X[X_obj] # dataframe으로 변경해주는것 

print(X_num.info())
print(X_obj.info())


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=True, handle_unknown='ignore')
# sparse : 희소행렬 생성 여부 -> 어느 행렬 자리가 1인지 확인 [메모리 낭비를 줄일 수 있음]
# handle_unknown : 새로운 내가 모르는 값이 들어오게 되면 어떻게 할지 정하는거 => 'ignore' default =  'error'

# 사이킷런의 전처리 클래스들은 학습용 클래스와 유사
# 1, fit 메소드
# 2, transform 메소드
# 3, fit_transform

X_obj_encoded = encoder.fit_transform(X_obj)
 
print(X_obj_encoded[:5]) # -> sparse = True 
# (0, 1)	1.0
# (0, 4)	1.0
# (1, 0)	1.0
# (1, 2)	1.0
# (2, 0)	1.0
# (2, 4)	1.0
# (3, 0)	1.0
# (3, 4)	1.0
# (4, 1)	1.0
# (4, 4)	1.0

# 주의사항
# 모든 사이킷런의 전처리 클래스는 transform numpy 배열로 변환


print(encoder.categories_)

print(encoder.feature_names_in_)


X_obj_encoded = pd.DataFrame(X_obj_encoded, columns=['s_f','s_m','e_C','e_Q','e_S'])



print(X_obj_encoded)

print(X_num.info())
print(X_obj_encoded.info())

X_num.reset_index(inplace = True)
X_obj_encoded.rest_index(inplace = True)

X = pd.concat([X_num, X_obj_encoded],axis = 1)
# concat을 하게 해주면 서로의 데이터가 추가가 되는데 빠진 데이터도 같이 결합이 되면 기존 데이터 보다 더 많게 된다.
# -> reset을 해준다



from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1, test_size=0.3)

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

model = RandomForestClassifier(n_estimators=100, max_depth=None, max_samples=1.0, class_weight='balanced',n_jobs=-1,random_state=1)
# max_depth = None 이면 train 
model.fit(X_train, y_train)

score = model.score(X_train, y_train)

print(f'Score(Train) : {score}')

score = model.score(X_test, y_test)
print(f'Score(Test) : {score}')


 