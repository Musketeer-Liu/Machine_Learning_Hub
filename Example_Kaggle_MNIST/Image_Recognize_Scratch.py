#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 18:48:59 2018

@author: yutong
"""


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout


num_classes = 10
size_images = 28
train_size = 30000
file_path = './train.csv'
data = pd.read_csv(file_path)


def process(raw):
    out_y = keras.utils.to_categorical(raw.label, num_classes)
    
    num_images = raw.shape[0]
    x_as_array = raw.values[:, 1:]
    x_shaped_array = x_as_array.reshape(num_images, size_images, size_images, 1)
    out_x = x_shaped_array/255

    return out_x, out_y

X, y = process(data)


model = Sequential()

model.add(Conv2D(30, kernel_size=(3,3), strides=2, activation='relu', 
                 input_shape=(size_images, size_images, 1)))
model.add(Dropout(0.5))
model.add(Conv2D(30, kernel_size=(3,3), strides=2, activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

model.fit(X, y, batch_size=120, epochs=2, validation_split=0.2)