# -*- coding: utf-8 -*-
"""
Created on Fri May 20 16:20:33 2022

@author: 오정민
"""
# para_grid가 list 로 들어가게 된다. 
# 조건부 매개변수에 맞는 값만 넣어주게 된다.
# 조건부 매개변수 기억하기! 문법 

# 성적을 올리기 위해서 : 전처리 처리를 먼저 해줘야 한다. -> 파라메타 조정 

# class_weight => 가중치 조절

# pipeline4 중요 -> 자료구조 queue = [작업의 순서를 지정해주는 변수]

from sklearn.datasets import load_breast_cancer

X, y = load_breast_cancer(return_X_y=True)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,stratify=y,random_state=11)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

base_model = LogisticRegression(n_jobs=1, random_state=11)

pipe = Pipeline([('s_scaler',scaler),
                 ('base_model',base_model)])

param_grid=[{'base_model__penalty' : ['l2'],
             'base_model__solver' : ['lbfgs'],
             'base_model__C' : [1.0,0.1,10,0.01,100],
             'base_modle__class_weight' : ['balanced',{0:0.9,1:0.1}]},
            {'base_model__penalty' : ['elasticnet'],
             'base_model__solver' : ['saga'],
             'base_model__C' : [1.0,0.1,10,0.01,100],
             'base_model_class_weight':
                 ['balanced',{0:0.9,1:0.1}]}]
    
cv = KFold(n_splits=5, shuffle=True, random_state=11)

grid_model = GridSearchCV(pipe,param_grid=param_grid,cv=cv,scoring='recall',n_jobs=1)

grid_model.fit(X_train,y_train)

print(f'best_score :{grid_model.best_score_}')
print(f'best_params : {grid_model.best_params_}')
print(f'best_model : {grid_model.best_estimator_}')


# 4, 학습된 머신러닝 모델의 평가
print(f'SCORE(TRAIN) : {grid_model.score(X_train, y_train)}')
print(f'SCORE(TEST) : {grid_model.score(X_test, y_test)}')

from sklearn.metrics import classification_report
pred_train = grid_model.predict(X_train)
pred_test = grid_model.predict(X_test)

print(classification_report(y_train, pred_train))
print(classification_report(y_test, pred_test))