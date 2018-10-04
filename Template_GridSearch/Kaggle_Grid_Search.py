#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 11:32:42 2018

@author: yutong
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor



file_path = './train.csv'
data = pd.read_csv(file_path)

def select(data):
    cols_low_cord = [col for col in data.columns if data[col].dtype=='object' and data[col].nunique()<10]
    cols_pure_num = [col for col in data.columns if data[col].dtype in ['int32', 'int64', 'float32', 'float64']]
    cols_selected = cols_low_cord + cols_pure_num
    return data[cols_selected]

data = pd.get_dummies(select(data))

features, label = data.drop(axis=1, columns=['SalePrice']), data.SalePrice
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.2, random_state=101)


estimators = []
estimators.append(('Impute', Imputer()))
estimators.append(('Model', XGBRegressor()))
pipeline = Pipeline(estimators)



def score(factor_1, factor_2, X_train, X_test, y_train, y_test):
    pipeline.set_params(Model__learning_rate = factor_1)
    pipeline.set_params(Model__n_estimators = factor_2)
    pipeline.fit(X_train, y_train)
    prediction = pipeline.predict(X_test)
    return mean_absolute_error(prediction, y_test)
    
    
print('Grid Search over 2 Factors: Learning Rate and Estimator Count')    
print('\n')

for factor_1 in [0.0001, 0.001, 0.01, 0.1, 1]:
    print('====================')
    for factor_2 in [100, 500, 1000, 5000, 10000]:
        mae = int(score(factor_1, factor_2, X_train, X_test, y_train, y_test))
        print('Learnning Rate: {} \t with Estimator Count: {} \t\t MAE Value: {}'.format(factor_1, factor_2, mae))
    print('\n')
    
    
    
    
    