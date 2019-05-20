# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 12:29:42 2018
Copyright (c) 2018 Mark B. Skouson

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""



import keras
from keras.datasets import mnist
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import RMSprop
import numpy as np

from myfunctions2 import *

Nkernels = 32
gpus = 1

#load the data
(x_train, y_train), (x_test, y_test)= mnist.load_data()

#turn the data to point clouds and create a feature vector for each point in the cloud.
maxpoints = max(find_max_points(x_train),find_max_points(x_test))

new_x_train = turn_to_point_centric(x_train,maxpoints)
new_x_test = turn_to_point_centric(x_test,maxpoints)
max3 = np.amax([np.amax(new_x_train),np.amax(new_x_test)])
new_x_train = 2*new_x_train/max3-1
new_x_test = 2*new_x_test/max3-1

new_y_train = keras.utils.to_categorical(y_train, 10)
new_y_test = keras.utils.to_categorical(y_test, 10)


# input shape is [total number of examples, number of points per example, dimensions of a point ]
# example for mnist [60000, 312, 2]



inputs = Input(shape=(maxpoints,new_x_train.shape[2]))
#these first layers are 'data augmentation' layers
x = MyAddScale(name='scale_augment')(inputs)
x = MyAdd2DRotation(name='rotate_augment')(x)
x = MyAddShift(name='shift_augment')(x)
x = MyAddJitter(name='jitter_augment')(x)
#This is the ursa layer to create a feature vector
x = MyUrsaMin(Nkernels,name='cluster')(x)
x = Activation('relu')(x)
x = BatchNormalization()(x)
#these last layers do classification
x = Dense(512,activation= 'relu',name='dense512')(x)
x = BatchNormalization()(x)
x = Dense(256,activation= 'relu',name='dense256')(x)
x = BatchNormalization()(x)
x = Dropout(rate=0.3)(x)
x = Dense(10,activation='softmax')(x)

model = Model(inputs=inputs, outputs= x)
if gpus>1:
    from keras.utils import multi_gpu_model
    model = multi_gpu_model(model, gpus=gpus)

rmsprop=tf.keras.optimizers.RMSprop(lr=.001, rho=.9,decay=.0001)
model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

history = model.fit(new_x_train, 
                    new_y_train, 
                    validation_data=(new_x_test,new_y_test),
                    epochs=500, batch_size=512, verbose=2)
