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
file_path = './train.csv'
data = pd.read_csv(file_path)


# Feature Engineering
deleted_col = ['Id', 'LotFrontage', 'MiscFeature', 'MiscVal', 'SalePrice']
selected_col = [col for col in data.columns if col not in deleted_col]
features = pd.get_dummies(data[selected_col])
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
feature_importance = gbt.feature_importances_
error = mean_squared_error(prediction, y_valid)
print(error)


# Construct Table
data = {'Importance Weight': feature_importance}
index = [X_train.columns]
feature_ranking = pd.DataFrame(data=data, index=index)


# Sort Table
feature_ranking = feature_ranking.sort_values(by=['Importance Weight'], ascending=False)


# Save Table
feature_ranking.to_csv(r'./feature_ranking.csv')