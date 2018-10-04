#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 17:34:42 2018

@author: yutong
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer
#from sklearn.impute import SimpleImputer

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#from xgboost import XGBRegressor

train_path = './train.csv'
test_path = './test.csv'


train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

#train.head()
#test.head()
#train.dtypes
#test.dtypes
#train.dtypes.value_counts()
#test.dtypes.value_counts()
#train.count().plot.bar(figsize=(12,3))
#test.count().plot.bar(figsize=(12,3))

cols_low_cord = [col for col in train.columns if train[col].nunique()<10 and train[col].dtype=='object']
cols_pure_num = [col for col in train.columns if train[col].dtype in ['float64', 'int64'] and col != 'SalePrice']
cols_selected = cols_low_cord + cols_pure_num

X_train = pd.get_dummies(train[cols_selected])
X_test = pd.get_dummies(test[cols_selected])
y_train = train.SalePrice

for col in list(set(X_test.columns) - set(X_train.columns)): X_train[col] = 0
for col in list(set(X_train.columns) - set(X_test.columns)): X_test[col] = 0

#pipeline = make_pipeline(SimpleImputer(), RandomForestRegressor())
pipeline = Pipeline([('Impute', Imputer()), ('Model', RandomForestRegressor())])
pipeline.fit(X_train, y_train)
prediction = pipeline.predict(X_test)
#print(prediction)

result = pd.DataFrame({'Id': test.Id, 'SalePrice': prediction})
#result.head()
result.to_csv('Result_RF_factor', index=False)