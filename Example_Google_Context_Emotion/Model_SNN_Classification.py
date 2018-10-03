#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 13:43:15 2018

@author: yutong
"""


# Import Packages
from __future__ import print_function
import collections, io, math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tensorflow as tf
from sklearn import metrics
from IPython import display

# Setup Format
tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format='{:.3f}'.format

# Load Data
train_url = 'https://storage.googleapis.com/mledu-datasets/sparse-data-embedding/train.tfrecord'
train_path = tf.keras.utils.get_file(train_url.split('/')[-1], train_url)
test_url = 'https://storage.googleapis.com/mledu-datasets/sparse-data-embedding/test.tfrecord'
test_path = tf.keras.utils.get_file(test_url.split('/')[-1], test_url)

informative_terms = ("bad", "great", "best", "worst", "fun", "beautiful",
                     "excellent", "poor", "boring", "awful", "terrible",
                     "definitely", "perfect", "liked", "worse", "waste",
                     "entertaining", "loved", "unfortunately", "amazing",
                     "enjoyed", "favorite", "horrible", "brilliant", "highly",
                     "simple", "annoying", "today", "hilarious", "enjoyable",
                     "dull", "fantastic", "poorly", "fails", "disappointing",
                     "disappointment", "not", "him", "her", "good", "time",
                     "?", ".", "!", "movie", "film", "action", "comedy",
                     "drama", "family", "man", "woman", "boy", "girl")

#terms_url = 'https://storage.googleapis.com/mledu-datasets/sparse-data-embedding/terms.txt'
#terms_path = tf.keras.utils.get_file(terms_url.split('/')[-1], terms_url)    
#informative_terms = None
#with io.open(terms_path, 'r', encoding='utf8') as f:
#    informative_terms = list(set(f.read().split()))


# Parse Template
def parse_template(data):
    feature = {'terms': tf.VarLenFeature(dtype=tf.string), 'labels': tf.FixedLenFeature(shape=[1], dtype=tf.float32)}
    feature_parsed = tf.parse_single_example(data, feature)    
    term, label = feature_parsed['terms'].values, feature_parsed['labels']
    return {'terms': term}, label

# Check Data
data_train = tf.data.TFRecordDataset(train_path)
data_test = tf.data.TFRecordDataset(test_path)
data_train = data_train.map(parse_template)
data_test = data_test.map(parse_template)

item = data_train.make_one_shot_iterator().get_next()
sess = tf.Session()
sess.run(item)


# Input Template
def input_template(filename, epoch_num=None, shuffle=True):
    data = tf.data.TFRecordDataset(filename)
    data = data.map(parse_template)
    if shuffle: data = data.shuffle(10000)
    
    # Feature Data
    data = data.padded_batch(25, data.output_shapes)
    data = data.repeat(epoch_num)
    
    feature, label = data.make_one_shot_iterator().get_next()   
    return feature, label


# Feature Columns with Information Terms
def column_template():
    return tf.feature_column.categorical_column_with_vocabulary_list(key='terms', vocabulary_list=informative_terms)

def embed_template(feature_columns, dimension):
    return tf.feature_column.embedding_column(feature_columns, dimension)

# Model Template
optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)    
optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)
hidden_units=[20,20]

dimension = 2
feature_columns = [embed_template(column_template(), dimension)]

train_input_fn = lambda: input_template([train_path])
test_input_fn = lambda: input_template([test_path])



model = tf.estimator.DNNClassifier(optimizer=optimizer, hidden_units=hidden_units, feature_columns=feature_columns)
model.train(input_fn=train_input_fn, steps=1000)
metrics_train = model.evaluate(input_fn=test_input_fn, steps=1000)
metrics_test = model.evaluate(input_fn=test_input_fn, steps=1000)

print('Train Metrics:')
for item in metrics_train:
    print(item, metrics_train[item])
print('--------------------------------------------')
print('Test Metrics:')
for item in metrics_test:
    print(item, metrics_test[item])
print('--------------------------------------------')


















