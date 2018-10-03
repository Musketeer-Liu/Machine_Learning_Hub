__author__ = 'Yutong Liu'
'''
2nd Step:
Use XGBoosting or GradientBoosting Algorithm 
Decide Features from selected ones with lower MSE compared to all features
'''


# Import Library
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor


ALL_ERROR = 729935083.1486999



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
    #'BsmtFullBath',
    #'2ndFlrSF',
    'Neighborhood',
    #'Functional',
    #'Exterior1st',
    #'BsmtFinSF2',
    #'MasVnrArea',
    'BsmtUnfSF',
    'ScreenPorch',
    'WoodDeckSF',
    #'MoSold'
    #'YrSold',
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


# Setup Model 
params = {
    'n_estimators': 1000,
    'learning_rate': 0.01,
    'max_depth': 3,         # default is 3
    'min_samples_split': 2, # default param
    'loss': 'ls'            # default param
}
gbt = GradientBoostingRegressor(**params)


# Train Model
gbt.fit(X_train, y_train)
prediction = gbt.predict(X_valid)


# Evaluate Model
error = mean_squared_error(prediction, y_valid)
print('Error with Full Columns: \t', error)
print('Error with Selected Columns \t', ALL_ERROR)
