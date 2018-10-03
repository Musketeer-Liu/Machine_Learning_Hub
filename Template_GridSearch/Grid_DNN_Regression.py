#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 16:50:33 2018

@author: yutong
"""


# Import Packages
import math
import numpy as np
import pandas as pd
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
#print('\nCorrelation Matrix Table: \n{}'.format(data_train.corr()))

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
def model_template(optimizer, feature_columns, 
                   steps, batch_size, hidden_units,                      
                   X_train, X_valid, y_train, y_valid):

    # Create Nerual Network Object
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
    dnn_regressor = tf.estimator.DNNRegressor(
            feature_columns=feature_columns, 
            hidden_units=hidden_units, 
            optimizer=optimizer)
    print('Model Created ', end='')
       
    # Create Input Function
    train_input_fn = lambda: input_template(X_train, y_train, batch_size=batch_size) 
    valid_X_input_fn = lambda: input_template(X_train, y_train, epoch_num=1, shuffle=False)
    valid_y_input_fn = lambda: input_template(X_valid, y_valid, epoch_num=1, shuffle=False)    
    
    # Start Model Training from Prior State
    dnn_regressor.train(input_fn=train_input_fn, steps=steps)
    print('==========> Model Triained ', end='')
    
    # Compute Prediction
    prediction_train = dnn_regressor.predict(input_fn=valid_X_input_fn)
    prediction_train = np.array([item['predictions'][0] for item in prediction_train])
    prediction_valid = dnn_regressor.predict(input_fn=valid_y_input_fn)
    prediction_valid = np.array([item['predictions'][0] for item in prediction_valid])
    print('==========> Model Predicted')
    print('----------------------------------------------------------------------------------------------------')
    
    # Compute RMSE Loss
    rmse_train = math.sqrt(metrics.mean_squared_error(prediction_train, y_train))
    rmse_valid = math.sqrt(metrics.mean_squared_error(prediction_valid, y_valid))
        
    # Return Result
    return dnn_regressor, rmse_train, rmse_valid




# Decode Help Function
def decode_float(data):
    return list(float(data[1:-1].split(',')[i].strip()) for i in range(len(data[1:-1].split(','))))

def decode_int(data):
    return list(int(data[1:-1].split(',')[i].strip()) for i in range(len(data[1:-1].split(','))))
    
def decode_help(data):
    return list(int(data.split(',')[i].strip()) for i in range(len(data.split(','))))

def decode_net(data):
    results = []
    for result in [item for item in data[1:-1].split('|')]:
        results.append(decode_help(result))
    return results

    
# Input Hyper-Parameters
print('Format for Hyper Parameters: Type Parameters in a List  | Separated by a Comma | Whitespace does not Matters')
print('Examples for Hyper Parameters: 1, 0.5, 0.1, 0.05, 0.01, 0.005 | 10, 50, 100, 500, 1000 | 1, 2,3, 4')
print('Format for Hidden Layers: Type Structure in a List | Separated by a Pipe | Whitespace does not Matter') 
print('Examples for Hidden Lyaers: [1,2,3|4,5,6|12,11] | [10,10,10] | [ 1,2, 3|4,5 ,6 | 12, 11 ]')

learning_rate_list = input('Please Decide Learning Rate List (Ex.: 1, 0.5, 0.1, 0.05, 0.01): ')
steps_list = input('Please Decide Steps List (Ex.: 10, 50, 100, 500, 1000): ')
batch_size_list = input('Please Decide Batch Size List: (Ex.: 10, 50, 100, 500 1000): ')
l1_regularization_strength_list = input('Please Decide L1 Regularization Strength (Ex.: 0, 0.1, 1): ')
hidden_units_list = input('Please Decide Hidden Units (Ex.: [3,3|5,5|10,10]): ')

# Decode Hyper-Parameters
#learning_rate_list = list(float(learning_rate_list.split(',')[i].strip()) for i in range(len(learning_rate_list.split(','))))
learning_rate_list = decode_float(learning_rate_list)
l1_regularization_strength_list = decode_float(l1_regularization_strength_list)
steps_list = decode_int(steps_list)
batch_size_list = decode_int(batch_size_list)
hidden_units_list = decode_net(hidden_units_list)

# Check Hyper-Parameters
print('\n\nHyper Parameter List Overview: ', end='')
print('Learning Rate: {} | Steps: {} | Batch Size: {} | L1 Regularization Strength: {}'.format(
        learning_rate_list, steps_list, batch_size_list, l1_regularization_strength_list))
print('Hidden Units Structure Overview: ', end='')
print('Hidden Units: {}'.format(hidden_units_list))

# Input Hyper-Parameters
#learning_rate = input('Please Decide Learning Rate: ')
#steps = input('Please Decide Steps: ')
#batch_size = input('Please Decide Batch Size: ')
#hidden_units = input('Please Decide Hidden Units: ')

# Decode Hyper-Parameters
#learning_rate = float(learning_rate)
#steps = int(steps)
#batch_size = int(batch_size)
#hidden_units = list(int(hidden_units.split(',')[i].strip()) for i in range(len(hidden_units.split(','))))


## Grid Search
print('====================================================================================================')
print('----------------------------------------------------------------------------------------------------')
a = 0
for hidden_units in hidden_units_list:
    a, b = a+1, 0
    for l1_regularization_strength in l1_regularization_strength_list:
        b, c = b+1, 0        
        for batch_size in batch_size_list:
            c, d = c+1, 0
            for steps in steps_list:
                d, e = d+1, 0
                for learning_rate in learning_rate_list: 
                    e = e+1           
            
                    test_num = str(a)+'.'+str(b)+'.'+str(c)+'.'+str(d)+'.'+str(e) 
        #            # Test Case on Test Dataset via Gradient Descent Optimizer
        #            model, rmse_train, rmse_valid = model_template(
        #                    optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate), 
        #                    feature_columns=column_template(normalize_data(preprocess_feature(data_train))), 
        #                    steps=steps, batch_size=batch_size, hidden_units=hidden_units, 
        #                    X_train=X_train, X_valid=X_valid, y_train=y_train, y_valid=y_valid)
                    
                    # Test Case on Test Dataset via FTRL Optimizer
                    model, rmse_train, rmse_valid = model_template(
                            optimizer=tf.train.FtrlOptimizer(learning_rate=learning_rate, 
                                                             l1_regularization_strength=l1_regularization_strength),
                            feature_columns=column_template(normalize_data(preprocess_feature(data_train))),
                            steps=steps, batch_size=batch_size, hidden_units=hidden_units,
                            X_train=X_train, X_valid=X_valid, y_train=y_train, y_valid=y_valid)
                    
        #            # Test Case on Test Dataset via Ada Delta Optimizer
        #            model, rmse_train, rmse_valid = model_template(
        #                    optimizer=tf.train.AdadeltaOptimizer(learning_rate=learning_rate),
        #                    feature_columns=column_template(normalize_data(preprocess_feature(data_train))),
        #                    steps=steps, batch_size=batch_size, hidden_units=hidden_units,
        #                    X_train=X_train, X_valid=X_valid, y_train=y_train, y_valid=y_valid) 
                    
        #            # Test Case on Test Dataset via Adam Optimizer
        #            model, rmse_train, rmse_valid = model_template(
        #                    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate),
        #                    feature_columns=column_template(normalize_data(preprocess_feature(data_train))),
        #                    steps=steps, batch_size=batch_size, hidden_units=hidden_units,
        #                    X_train=X_train, X_valid=X_valid, y_train=y_train, y_valid=y_valid)            
                    
                    data_test = pd.read_csv('https://storage.googleapis.com/mledu-datasets/california_housing_test.csv', sep=',')
                    
                    X_test = normalize_data(preprocess_feature(data_test))
                    y_test = preprocess_label(data_test)
                    
                    test_input_fn = lambda: input_template(X_test, y_test, epoch_num=1, shuffle=None)
                    
                    prediction_test = model.predict(input_fn=test_input_fn)
                    prediction_test = np.array([item['predictions'][0] for item in prediction_test])
                    
                    rmse_test = math.sqrt(metrics.mean_squared_error(prediction_test, y_test))
                    
                    
                    print('Parameter:  {}  ||  Learning Rate: {} | Steps: {} | Batch Size: {} | L1: {} | Hidden Units: {}'.format(
                            test_num, learning_rate, steps, batch_size, l1_regularization_strength, str(hidden_units)))
                    print('Evaluation: {}  ||  RMSE Train: {:.3f} | RMSE Valid: {:.3f} | RMSE Test: {:.3f}'.format(
                            test_num, rmse_train, rmse_valid, rmse_test))
                    print('----------------------------------------------------------------------------------------------------')
                print('====================================================================================================')
            print('----------------------------------------------------------------------------------------------------\n')
        print('====================================================================================================\n')
    print('====================================================================================================\n\n')
print('Grid Search Finished!\n')

