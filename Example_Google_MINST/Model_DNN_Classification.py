#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 11 13:19:10 2018

@author: yutong
"""


# Import Packages
import glob, math, os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import cm, gridspec
import seaborn as sns
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

# Setup Format
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows=10
pd.options.display.float_format = '{:.3f}'.format

# Load Data
data_train = pd.read_csv('./mnist_train_small.csv', sep=',', header=None)
data_train = data_train.head(10000)
data_train = data_train.reindex(np.random.permutation(data_train.index))
data_test = pd.read_csv('./mnist_test.csv', sep=',', header=None)

# Check Data
#print(data_train.head())
#print(data_train.describe())


# Pre-Process Data
def preprocess_data(data):
    feature = data.loc[:, 1:784]/255
    label = data[0]
    return feature, label

# Split Train-Valid Data
X_train, y_train = preprocess_data(data_train[:7500])
X_valid, y_valid = preprocess_data(data_train[7500:10000])


# Construct Feature Column 
def column_template():
    return set([tf.feature_column.numeric_column('pixels', shape=784)])
    
# Template Train Function
def train_template(feature, label, batch_size, epoch_num=None, shuffle=True):
    def _input_fn(epoch_num=None, shuffle=True):
        idx = np.random.permutation(feature.index)
        feature_new = {'pixels': feature.reindex(idx)}
        label_new = np.array(label[idx])
    
        ds = Dataset.from_tensor_slices((feature_new, label_new))
        ds = ds.batch(batch_size).repeat(epoch_num)
        if shuffle: ds = ds.shuffle(10000)
        
        feature_next, label_next = ds.make_one_shot_iterator().get_next()
        return feature_next, label_next
    
    return _input_fn
    
# Template Valid Function
def valid_template(feature, label, batch_size):
    def _input_fn():
        feature_new = {'pixels': feature.values}
        label_new = np.array(label)
        
        ds = Dataset.from_tensor_slices((feature_new, label_new))
        ds = ds.batch(batch_size)
    
        feature_next, label_next = ds.make_one_shot_iterator().get_next()
        return feature_next, label_next
    
    return _input_fn

# Template Model Object
def model_template(optimizer, steps, batch_size, hidden_units, feature_columns,
                   X_train, X_valid, y_train, y_valid):
    n_classes = 10
    periods = 10
    steps_per_period = steps / periods
    config = tf.contrib.learn.RunConfig(keep_checkpoint_max=1)
    
    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
    dnn_classifier = tf.estimator.DNNClassifier(
            feature_columns=feature_columns, n_classes=n_classes, 
            hidden_units=hidden_units, optimizer=optimizer, config= config)

    train_input_fn = train_template(X_train, y_train, batch_size=batch_size)
    valid_X_input_fn = valid_template(X_train, y_train, batch_size=batch_size)
    valid_y_input_fn = valid_template(X_train, y_train, batch_size=batch_size)
    
    print('Model Train Initiating...')
    print('Log Loss during Training:')
    log_losses_train, log_losses_valid = [], []
    
    for period in range(0, periods):
        dnn_classifier.train(input_fn=train_input_fn, steps=steps_per_period)
        
        prediction_train = list(dnn_classifier.predict(input_fn=valid_X_input_fn))
        prediction_train = np.array([item['probabilities'] for item in prediction_train])
        class_id_train = np.array([item['class_ids'][0] for item in prediction_train])
        encoding_train = tf.keras.utils.to_categorical(class_id_train, 10)
        
        prediction_valid = list(dnn_classifier.predict(input_fn=valid_y_input_fn))
        prediction_valid = np.array([item['probabilities'] for item in prediction_valid])
        class_id_valid = np.array([item['class_ids'][0] for item in prediction_valid])
        encoding_valid = tf.keras.utils.to_categorical(class_id_valid, 10)
        
        log_loss_train = metrics.log_loss(y_train, encoding_train)
        log_losses_train.append(log_loss_train)
        log_loss_valid = metrics.log_loss(y_valid, encoding_valid)
        log_losses_valid.append(log_loss_valid)
        
        print('Period_{:02d} | Train Loss: {:.3f} | Valid Loss: {:.3f}'.format(period, log_loss_train, log_loss_valid))
        
    print('Model Train Finished!')
    
#    _ = map(os.remove, glob.glob(os.path.join(dnn_classifier.model_dir, 'events.out.tfevents*')))
    
    accuracy_train = metrics.accuracy_score(y_train, class_id_train)
    accuracy_valid = metrics.accuracy_score(y_valid, class_id_valid)
    print('Final Train Loss: \t{:.3f} | Final Train Accuracy: \t{:.3f}'.format(log_loss_train, accuracy_train))
    print('Final Valid Loss: \t{:.3f} | Final Valid Accuracy: \t{:.3f}'.format(log_loss_valid, accuracy_valid))
    
    plt.ylabel("LogLoss")
    plt.xlabel("Periods")
    plt.title("LogLoss vs. Periods")
    plt.plot(log_loss_train, label="training")
    plt.plot(log_loss_valid, label="validation")
    plt.legend()
    plt.show()
  
    # Output a plot of the confusion matrix.
    cm = metrics.confusion_matrix(y_valid, class_id_valid)
    # Normalize the confusion matrix by row (i.e by the number of samples
    # in each class)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    ax = sns.heatmap(cm_normalized, cmap="bone_r")
    ax.set_aspect(1)
    plt.title("Confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.show()   
    
    return dnn_classifier


# Test Cases
model = model_template(optimizer=tf.train.AdagradOptimizer(learning_rate=0.05),
                       steps = 1000, batch_size = 10, hidden_units = [100, 100],
                       feature_columns = column_template(),
                       X_train=X_train, X_valid=X_valid, y_train=y_train, y_valid=y_valid)

X_test, y_test = preprocess_data(data_test)
test_input_fn = valid_template(X_test, y_test, batch_size=100)

prediction_test = model.predict(input_fn=test_input_fn)
prediction_test = np.array([item['probabilities'] for item in prediction_test])
class_id_test = np.array([item['class_ids'][0] for item in prediction_test])
encoding_test = tf.keras.utils.to_categorical(class_id_test, 10)

log_loss_test = metrics.log_loss(y_test, encoding_test)
accuracy_test = metrics.accuracy_score(y_test, class_id_test)

print('Final Test Loss: \t{:.3f} | Final Test Accuracy: \t{:.3f}'.format(log_loss_test, accuracy_test))








 