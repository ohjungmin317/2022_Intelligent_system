# -*- coding: utf-8 -*-
"""
Created on Fri May 20 15:04:14 2022

@author: 오정민
"""
# 머신러닝 -> 해당 알고리즘을 명확하게 이해하고 데이터 특징에 대해 이해하고 내것으로 
# 바꾸면 된다.

# 1번
# score = model.score(X_test, y_test) -> 논리적으로 잘못된 부분(평가기준을 잘못잡음)
# [test -> 최종 데이터에 대해 평가하는 것]

# 위의 데이터를 예방하기 위해서 -> 검증데이터


# 2번
# score = model.score(X_valid, y_valid) -> 이론상은 best -> 실전에서는 worst
# 문제인 이유? -> 데이터가 분할이 어떻게 되느냐에 따라서 값이 달라진다.  


import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 교차검증 데이터 셋을 분할하기 위한 클래스
from sklearn.model_selection import KFold
# 교차검증을 수행할 수 있는 함수
from sklearn.model_selection import cross_val_score

from sklearn.ensemble import GradientBoostingClassifier

X, y = load_iris(return_X_y=True)

print(X.shape)
print(y.shape)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1,stratify=y)

# 모델의 학습에 사용할 파라메터의 정의

param_grid = {'learning_rate':[0.1,0.2,0.3,1,0.01],
              'max_depth':[1,2,3],
              'n_estimators':[100,200,300,10,50],
              'subsample':[1.,0.8,0.5,0.3,0.1]}
# 교차 검증 점수를 기반으로 최적의 하이퍼 파라메터를 검색할 수 있는 GridSearchCV클래스

from sklearn.model_selection import GridSearchCV

# 교차 검증 수행을 위한 데이터 분할 객체
cv = KFold(n_splits=5, shuffle=True,random_state=1)
# 교차 검증에 사용할 기본 머신러닝 모델
# - 테스트할 하이퍼 파라메터는 설정에서 제외
# - 공통 하이퍼파라메터 정보만 사용하여 모델을 정의
# - 학습 x

base_model = GradientBoostingClassifier(random_state=1)

# GridSearchCV 클래스의 하이퍼 파라메터 정보
# GridSearchCV( 예측기 객체, 테스트 파라메터의 딕셔너리 객체, cv= 교차검증 폴드 수, ...)

grid_model = GridSearchCV(estimator=base_model, param_grid=param_grid,cv=cv,n_jobs=1,verbose=3,scoring='precision')

grid_model.fit(X_train, y_train)

# 모든 하이퍼 파라메터를 조합하여 평가한 가장 높은 교차검증 SCORE를 반환
print(f'best_score -> {grid_model.best_score_}')
print(f'best_params ->{grid_model.best_params_}')
print(f'best_model ->{grid_model.best_estmator_}')

score = grid_model.score(X_train, y_train)
print(f'SCORE(TRAIN) : {score:.5f}')
score = grid_model.score(X_test, y_test)
print(f'SCORE(TEST) : {score:.5f}')