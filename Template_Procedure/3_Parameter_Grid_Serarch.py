__author__ = 'Yutong Liu'
'''
3rd Step:
Use XGBoosting or GradientBoosting Algorithm 
Grid search best parameters to optimize model
'''


# Import Library
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor




# Load Data
file_path = './train.csv'
data = pd.read_csv(file_path)


# Feature Engineering
decided_col = [
    'OverallQual',
    'GrLivArea',
    '1stFlrSF',
    'YearRemodAdd',
    'BsmtFinSF1',
    'LotArea',
    'TotalBsmtSF',
    'YearBuilt',
    'GarageArea',
    'OverallCond',
    'Fireplaces',
    'OpenPorchSF',
    'GarageCars',
    'GarageYrBlt',
    'Neighborhood',
    'BsmtUnfSF',
    'ScreenPorch',
    'WoodDeckSF',
]
features = pd.get_dummies(data[decided_col])
label = data['SalePrice'] - data['MiscVal']


# Data Cleaning
for col in features.columns:
    if np.issubdtype(features[col].dtype, np.number):
        ixes = np.isnan(features[col])
        # # Get to know which numerica column has NAN data:
        # if len(np.unique(ixes)) == 2: print(col)
        features.loc[ixes, col] = 0


# Split Data
X_train, X_valid, y_train, y_valid = train_test_split(features, label, test_size=0.2, random_state=101)


# Grid Search 
for n_estimators in [500, 750, 1000, 1500]:
    for learning_rate in [0.1, 0.05, 0.02, 0.01]:
        for max_depth in [3, 6]:
            params = {
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'max_depth': max_depth,
                'random_state': 101
            }
            gbt = GradientBoostingRegressor(**params)
            
            gbt.fit(X_train, y_train)
            prediction = gbt.predict(X_valid)
            error = mean_squared_error(prediction, y_valid)
            print('Estimator #: {} | Learning Rate: {} | Max Depth: {} ==> \tError: {}'.format(
                n_estimators, learning_rate, max_depth, error
            ))
