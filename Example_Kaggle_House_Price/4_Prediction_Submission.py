__author__ = 'Yutong Liu'
'''
1st Step:
Use XGBoosting or GradientBoosting Algorithm 
Record MSE with all feature columns for future comparison
Select Features via Factor Importance
'''


# Import Library
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor




# Load Data
train_file_path = './train.csv'
train_data = pd.read_csv(train_file_path)
test_file_path = './test.csv'
test_data = pd.read_csv(test_file_path)


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
train_features = pd.get_dummies(train_data[decided_col])
train_label = train_data['SalePrice'] - train_data['MiscVal']
test_features = pd.get_dummies(test_data[decided_col])


# Data Cleaning
for col in train_features.columns:
    if np.issubdtype(train_features[col].dtype, np.number):
        ixes = np.isnan(train_features[col])
        train_features.loc[ixes, col] = 0
for col in test_features.columns:
    if np.issubdtype(test_features[col].dtype, np.number):
        ixes = np.isnan(test_features[col])
        test_features.loc[ixes, col] = 0


# Create Model
params = {
    'n_estimators': 1500,
    'learning_rate': 0.05,
    'max_depth': 3
}
gbt = GradientBoostingRegressor(**params)


# Train Model
gbt.fit(train_features, train_label)
test_prediction = gbt.predict(test_features)
test_prediction = test_prediction + test_data['MiscVal']


# Construct Table
data = {'Id': test_data.Id, 'SalePrice': test_prediction}
result = pd.DataFrame(data=data)


# Save Table
result.to_csv(r'./HOUSE_PRICE_PREDICTION.csv', index=False)