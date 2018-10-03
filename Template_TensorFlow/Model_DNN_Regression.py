#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 22:55:38 2018

@author: yutong
"""


# Import Packages
import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.python.data import Dataset
from sklearn import metrics

# Setup Format
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.3f}'.format

# Load Data
data_train = pd.read_csv('https://storage.googleapis.com/mledu-datasets/california_housing_train.csv', sep=',')
data_train = data_train.reindex(np.random.permutation(data_train.index))

# Check Data
#data_train.describe()
#data_train.hist(bins=20)
print('\nCorrelation Matrix Table: \n{}'.format(data_train.corr()))

# Pre-Process Features
def preprocess_feature(data):
    # Select Original Feature
    feature_selected = data[['longitude', 'latitude', 'housing_median_age', 'total_rooms',
                             'total_bedrooms', 'population', 'households', 'median_income']]
    # Create Synthetic Feature
    feature_processed = feature_selected.copy()
    feature_processed['rooms_per_person'] = data['total_rooms'] / data['population']
    return feature_processed

# Pre-Process Label
def preprocess_label(data):
    label_processed = pd.DataFrame()
    label_processed['median_house_value'] = data['median_house_value'] / 1000.0
    return label_processed

# Linear Scale Function
def linear_scale(series):
    return series.apply(lambda x:((x-series.min())*2.0/(series.max()-series.min()))-1.0)

# Option -- Log Normalize Scale Function
def log_normalize(series):
    return series.apply(lambda x: math.log(x+1.0))

# Option -- Clip Scale Function
def clip_scale(series, clip_min, clip_max):
    return series.apply(lambda x:(min(max(x, clip_min), clip_max)))

# Option -- Z Score Normalize Scale Function
def z_score_normalize(series):
    mean, std = series.mean(), series.std()
    return series.apply(lambda x: (x-mean)/std)

# Option -- Binary Threshold Scale Function
def binary_threshold(series, threshold):
    return series

## Normalize Data with Linear Scale Function
#def normalize_data(data):
#    data_normalized = pd.DataFrame()
#    for item in data.columns:
#        data_normalized[item] = linear_scale(data[item])
#    return data_normalized  

# Normalize Data with Multiple Scale Function
def normalize_data(data):
    data_normalized = pd.DataFrame()
    data_normalized['longitude'] = linear_scale(data['longitude'])
    data_normalized['latitude'] = linear_scale(data['latitude'])
    data_normalized['housing_median_age'] = linear_scale(data['housing_median_age'])
    
    data_normalized['total_bedrooms'] = log_normalize(data['total_bedrooms'])    
    data_normalized['households'] = log_normalize(data['households'])
    data_normalized['median_income'] = log_normalize(data['median_income'])
    
    data_normalized['total_rooms'] = linear_scale(clip_scale(data['total_rooms'], 0, 10000))
    data_normalized['population'] = linear_scale(clip_scale(data['population'], 0, 5000))
    data_normalized['rooms_per_person'] = linear_scale(clip_scale(data['rooms_per_person'], 0, 5))
    
    return data_normalized


# Split Train-Test Dataset
X_train = normalize_data(preprocess_feature(data_train.head(12000)))
X_valid = normalize_data(preprocess_feature(data_train.tail(5000)))
y_train = preprocess_label(data_train.head(12000))
y_valid = preprocess_label(data_train.tail(5000))
#print('\nTraining Features (X_train) Summary: \n{}'.format(X_train.describe()))
#print('\nValidation Featurews (X_valid) Summary: \n{}'.format(X_valid.describe()))                                            
#print('\nTraining Label (y_train) Summary: \n{}'.format(y_train.describe())
#print('\nValidation Label (y_valid) Summary \n{}:'.format(y_valid.describe()))


# Construct TensorFlow Feature Columns
def column_template(data):
    return set([tf.feature_column.numeric_column(item) for item in data])

# Template Input Function
def input_template(feature, label, batch_size=1, epoch_num=None, shuffle=True):
    """Return: A Tuple (feature, label) for next data batch"""
    # Convert pandas data into a dict of np arrays
    feature = {key: np.array(value) for key,value in dict(feature).items()}
    # Construct a dataset with configured Batch, Epoch, and Shuffle
    ds = Dataset.from_tensor_slices((feature, label))
    ds = ds.batch(batch_size).repeat(epoch_num)
    if shuffle: ds = ds.shuffle(10000)
    # Return the next batch of data
    feature, label = ds.make_one_shot_iterator().get_next()
    return feature, label

# Template Training model
def model_template(optimizer, steps, batch_size, hidden_units, feature_columns,                     
                   X_train, X_valid, y_train, y_valid):
    # Setup Initial Data
    periods = 10
    steps_per_period = steps / periods
    
    # Create Nerual Network Object
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
    dnn_regressor = tf.estimator.DNNRegressor(
            feature_columns=feature_columns, 
            hidden_units=hidden_units, 
            optimizer=optimizer)
       
    # Create Input Function
    train_input_fn = lambda: input_template(X_train, y_train, batch_size=batch_size) 
    valid_X_input_fn = lambda: input_template(X_train, y_train, epoch_num=1, shuffle=False)
    valid_y_input_fn = lambda: input_template(X_valid, y_valid, epoch_num=1, shuffle=False)    
    
    # Train Model with Loss Metrics in a loop
    print('\nModel Training Initiating...')
    print('RMSE of Training Data:')
    rmses_train, rmses_valid = [], []
    
    for period in range(0, periods):
        # Start Model Training from Prior State
        dnn_regressor.train(input_fn=train_input_fn, steps=steps_per_period)
        
        # Compute Prediction
        prediction_train = dnn_regressor.predict(input_fn=valid_X_input_fn)
        prediction_train = np.array([item['predictions'][0] for item in prediction_train])
        prediction_valid = dnn_regressor.predict(input_fn=valid_y_input_fn)
        prediction_valid = np.array([item['predictions'][0] for item in prediction_valid])
        
        # Compute RMSE Loss
        rmse_train = math.sqrt(metrics.mean_squared_error(prediction_train, y_train))
        rmses_train.append(rmse_train)
        rmse_valid = math.sqrt(metrics.mean_squared_error(prediction_valid, y_valid))
        rmses_valid.append(rmse_valid)
        
        # Print Current Loss
        print('Period_{:02d} | Train: {:.3f} | Valid: {:.3f}'.format(period, rmse_train, rmse_valid))
        
    print('Model Training Finished!\n')    
    
    # Plot Loss Metrics over Periods
    plt.title('RMSE vs. Periods')
    plt.xlabel('Periods')
    plt.ylabel('RMSE')
    plt.tight_layout()
    plt.plot(rmses_train, label='RMSE Train')
    plt.plot(rmses_valid, label='RMSE Valid')
    plt.legend(loc=1)
        
    # Return Result
    print('Final RMSE on Training Data: \t{:.3f}'.format(rmse_train))
    print('Final RMSE on Validation Data: \t{:.3f}'.format(rmse_valid))
    return dnn_regressor


## Test Case on Test Dataset
#model = model_template(optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.05), 
#                  feature_columns=construct_feature_columns(X_train), 
#                  steps=500, batch_size=50, hidden_units=[10,10,10], 
#                  X_train=X_train, X_valid=X_valid, y_train=y_train, y_valid=y_valid)

## Test Case on Test Dataset via FTRL Optimizer
#model = model_template(optimizer=tf.train.FtrlOptimizer(learning_rate=0.05),
#                   feature_columns=construct_feature_columns(X_train),
#                   steps=1000, batch_size=100, hidden_units=[10,10,10],
#                   X_train=X_train, X_valid=X_valid, y_train=y_train, y_valid=y_valid)

## Test Case on Test Dataset via Ada Delta Optimizer
#model = model_template(optimizer=tf.train.AdadeltaOptimizer(learning_rate=1),
#                   feature_columns=construct_feature_columns(X_train),
#                   steps=500, batch_size=100, hidden_units=[10,10,10],
#                   X_train=X_train, X_valid=X_valid, y_train=y_train, y_valid=y_valid)

# Test Case on Test Dataset via Adam Optimizer
model = model_template(optimizer=tf.train.AdamOptimizer(learning_rate=0.005),
                  feature_columns=column_template(normalize_data(preprocess_feature(data_train))),
                  steps=500, batch_size=50, hidden_units=[10,10,10],
                  X_train=X_train, X_valid=X_valid, y_train=y_train, y_valid=y_valid)

data_test = pd.read_csv('https://storage.googleapis.com/mledu-datasets/california_housing_test.csv', sep=',')

X_test = normalize_data(preprocess_feature(data_test))
y_test = preprocess_label(data_test)

test_input_fn = lambda: input_template(X_test, y_test, epoch_num=1, shuffle=None)

prediction_test = model.predict(input_fn=test_input_fn)
prediction_test = np.array([item['predictions'][0] for item in prediction_test])

rmse_test = math.sqrt(metrics.mean_squared_error(prediction_test, y_test))
print('Final RMSE on Testing Data: \t{:.3f}'.format(rmse_test))
