#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 15:57:49 2018

@author: yutong
"""
"""
ROC & AUC

"""

# Import Packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.python.data import Dataset
from sklearn import metrics

# Setup Format
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows=10
pd.options.display.float_format='{:.3f}'.format

# Load Data
data_train = pd.read_csv('https://storage.googleapis.com/mledu-datasets/california_housing_train.csv', sep=',')
data_train.reindex(np.random.permutation(data_train.index))

# Check Data
#data_train.describe()
print('\nCorrelation Matrix Table: \n{}'.format(data_train.corr()))


# Pre-Process Features
def preprocess_feature(data):
    feature_selected = data[['latitude','longitude','housing_median_age','total_rooms',
                             'total_bedrooms','population','households','median_income']]
    feature_processed = feature_selected.copy()
    feature_processed['rooms_per_person'] = data['total_rooms'] / data['population']
    return feature_processed

# Pre-Process Label
def preprocess_label(data):
    label_processed = pd.DataFrame()
    label_processed['high_house_value'] = (data['median_house_value']>265000).astype(float)
    return label_processed


# Bucketize Data
def quantize_boundary(feature_value, bucket_num):
    boundary = np.arange(1.0, bucket_num) / bucket_num
    quantile = feature_value.quantile(boundary)
    return [quantile[i] for i in quantile.keys()]

## Construct TensorFlow Feature Columns
#def construct_feature_columns(data):
#    return set([tf.feature_column.numeric_column(item) for item in data])

# Construct TensorFlow Feature Columns
def column_template(data):
    """
    Return: A set of processed feature colunms
    """    
    # Transfer Feature Columns to Numeric Columns
    longitude = tf.feature_column.numeric_column('longitude')
    latitude = tf.feature_column.numeric_column('latitude')
    housing_median_age = tf.feature_column.numeric_column('housing_median_age')
    total_rooms = tf.feature_column.numeric_column('total_rooms')
    total_bedrooms = tf.feature_column.numeric_column('total_bedrooms')
    population = tf.feature_column.numeric_column('population')
    households = tf.feature_column.numeric_column('households')
    median_income = tf.feature_column.numeric_column('median_income')
    rooms_per_person = tf.feature_column.numeric_column('rooms_per_person')
    
    # Bucketize the Numeric Columns
    longitude_bucket = tf.feature_column.bucketized_column(
            longitude, boundaries=quantize_boundary(data['longitude'], 10))
    latitude_bucket = tf.feature_column.bucketized_column(
            latitude, boundaries=quantize_boundary(data['latitude'], 10))
    housing_median_age_bucket = tf.feature_column.bucketized_column(
            housing_median_age, boundaries=quantize_boundary(data['housing_median_age'], 5))
    total_rooms_bucket = tf.feature_column.bucketized_column(
            total_rooms, boundaries=quantize_boundary(data['total_rooms'], 5))
    total_bedrooms_bucket = tf.feature_column.bucketized_column(
            total_bedrooms, boundaries=quantize_boundary(data['total_bedrooms'], 5))
    population_bucket = tf.feature_column.bucketized_column(
            population, boundaries=quantize_boundary(data['population'], 5))
    households_bucket = tf.feature_column.bucketized_column(
            households, boundaries=quantize_boundary(data['households'], 5))
    median_income_bucket = tf.feature_column.bucketized_column(
            median_income, boundaries=quantize_boundary(data['median_income'], 5))
    rooms_per_person_bucket = tf.feature_column.bucketized_column(
            rooms_per_person, boundaries=quantize_boundary(data['rooms_per_person'], 5))
    
    # Create Crossed Feature Columns
    long_x_lat = tf.feature_column.crossed_column(
            set([longitude_bucket, latitude_bucket]), hash_bucket_size=50)
    
    # Zip the result into a set
    result = set([longitude_bucket, latitude_bucket, housing_median_age_bucket,
                  total_rooms_bucket, total_bedrooms_bucket, population_bucket,
                  households_bucket, median_income_bucket, rooms_per_person_bucket,
                  long_x_lat])    
    return result   


# Split Train-Test Dataset
X_train = preprocess_feature(data_train.head(12000))
X_valid = preprocess_feature(data_train.tail(5000))
y_train = preprocess_label(data_train.head(12000))
y_valid = preprocess_label(data_train.tail(5000))
#print('\nTraining Features (X_train) Summary: \n{}'.format(X_train.describe()))
#print('\nValidation Features (X_valid) Summary: \n{}'.format(X_valid.describe()))
#print('\nTraining Label (y_train) Summary: \n{}'.format(y_train.describe()))
#print('\nValidation Label (y_valid) Summary: \n'.format(y_valid.describe()))
 

# Model Size help function -- Only for FTRL Optimizer
# Characterize l1_regulation_strength parameter
def model_size(estimator):
  variables = estimator.get_variable_names()
  size = 0
  for variable in variables:
    if not any(x in variable for x in ['global_step','centered_bias_weight', 'bias_weight', 'Ftrl']):
      size += np.count_nonzero(estimator.get_variable_value(variable))
  return size


# Template Input Function
def input_template(feature, label, batch_size=1, epoch_num=None, shuffle=True):
    """
    Return: A Tuple of (feature, label) for next data batch
    """
    # Convert pandas data into a dict of np arrays
    feature = {key: np.array(value) for key,value in dict(feature).items()}    
    # Construct a dataset with figured Batch, Epoch and Shffule
    ds = Dataset.from_tensor_slices((feature, label))
    ds = ds.batch(batch_size).repeat(epoch_num)
    if shuffle: ds = ds.shuffle(10000)    
    # Return the next batch of data
    feature, label = ds.make_one_shot_iterator().get_next()
    return feature, label

# Template Training Model
def model_template(optimizer, steps, batch_size, feature_columns,
                   X_train, X_valid, y_train, y_valid):
    # Setup Initial Data
    periods = 10
    steps_per_period = steps / periods
    
    # Create Linear Classifier Object
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
    linear_classifier = tf.estimator.LinearClassifier(feature_columns=feature_columns, optimizer=optimizer)
    
    # Create Input Functions
    train_input_fn = lambda: input_template(X_train, y_train, batch_size=batch_size)
    valid_X_input_fn = lambda: input_template(X_train, y_train, epoch_num=1, shuffle=False)
    valid_y_input_fn = lambda: input_template(X_valid, y_valid, epoch_num=1, shuffle=False)
    
    # Train Model - Predict Loss in a loop
    print('\nModel Training Initiating...')
    print('LogLoss of Training Data:')
    log_losses_train, log_losses_valid = [], []
    
    for period in range(0, periods):
        # Start Model Training from Prior State
        linear_classifier.train(input_fn=train_input_fn, steps=steps_per_period)
        
        # Compute Predictions
        probability_train = linear_classifier.predict(input_fn=valid_X_input_fn)
        probability_train = np.array([item['probabilities'] for item in probability_train])
        probability_valid = linear_classifier.predict(input_fn=valid_y_input_fn)
        probability_valid = np.array([item['probabilities'] for item in probability_valid])
        evaluation_metrics = linear_classifier.evaluate(input_fn=valid_y_input_fn)
        auc, accu = evaluation_metrics['auc'], evaluation_metrics['accuracy']
        
        # Compute RMSE Loss
        log_loss_train = metrics.log_loss(y_train, probability_train)
        log_losses_train.append(log_loss_train)
        log_loss_valid = metrics.log_loss(y_valid, probability_valid)
        log_losses_valid.append(log_loss_valid)
        
        # Print Current Loss
        print('Period_{:02d} | Log_Loss: {:.3f} | AUC: {:.3f} | Accuracy: {:.3f}'.format(period, log_loss_train, auc, accu))
    
    print('Model Training Finished! \n')
    print('')
    
    # Plot Loss Metrics over Periods
    plt.figure(figsize=(10, 4))
    
    plt.subplot(121)
    plt.title('LogLoss vs. Periods')
    plt.xlabel('Periods')
    plt.ylabel('LogLoss')
    plt.tight_layout()
    plt.plot(log_losses_train, label='Log Loss Train')
    plt.plot(log_losses_valid, label='Log Loss Valid')
    plt.legend(loc=1)
    
    # Plot ROC Curve
    probability_roc = linear_classifier.predict(input_fn=valid_y_input_fn)
    probability_roc = np.array([item['probabilities'][1] for item in probability_roc])
    false_rate, true_rate, threshold = metrics.roc_curve(y_valid, probability_roc)
    
    plt.subplot(122)
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.plot(false_rate, true_rate, label='Our Calssifier')
    plt.plot([0,1], [0,1], label='Random Classifier')
    plt.legend(loc = 2)
        
    # Return Result
    print('Final Log Loss of Training Data: \t{:.3f}'.format(log_loss_train))
    print('Final Log Loss of Validation Data: \t{:.3f}'.format(log_loss_valid))
    return linear_classifier
    



# Test Case on Test Dataset via Linear Classifier Object
model = model_template(optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.05),
                       steps=500, batch_size=100, 
                       feature_columns=column_template(preprocess_feature(data_train)),
                       X_train=X_train, X_valid=X_valid, y_train=y_train, y_valid=y_valid)

## Test Case on Test Dataset via Ada Delta Optimizer
#model = model_template(optimizer = tf.train.AdadeltaOptimizer(learning_rate=125), 
#                       steps=500, batch_size=500, 
#                       feature_columns=column_template(preprocess_feature(data_train)),
#                       X_train=X_train, X_valid=X_valid, y_train=y_train, y_valid=y_valid)

## Test Case on Test Dataset via Adam Optimizer
#model = model_template(optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1), 
#                       steps=500, batch_size=500,
#                       feature_columns=column_template(preprocess_feature(data_train)),
#                       X_train=X_train, X_valid=X_valid, y_train=y_train, y_valid=y_valid)

## Test Case on Test Dataset via FTRL Optimizer
#l1_strength = 0.1 # Only for FTRL Optimizer
#model = model_template(optimizer=tf.train.FtrlOptimizer(learning_rate=0.1,
#                                                          l1_regularization_strength=l1_strength),
#                       steps=500, batch_size=500,
#                       feature_columns=column_template(preprocess_feature(data_train)),
#                       X_train=X_train, X_valid=X_valid, y_train=y_train, y_valid=y_valid)

data_test = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_test.csv", sep=",")
X_test = preprocess_feature(data_test)
y_test = preprocess_label(data_test)

test_input_fn = lambda: input_template(X_test, y_test, epoch_num=1, shuffle=False)
probability_test = model.predict(input_fn=test_input_fn)
probability_test = np.array([item['probabilities'] for item in probability_test])
log_loss_test = metrics.log_loss(y_test, probability_test)
#print('\nL1 Strength: {} | Model Size: {}'.format(l1_strength, model_size(model))) # Only for FTRL Optimizer
print('Final Log Loss of Testing Data: \t{:.2f}'.format(log_loss_test))