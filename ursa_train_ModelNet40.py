# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 09:34:57 2018

@author: mskouson
"""

"""
this script is to train an URSA layer network on MOdelNet40 data
usage: 
python ursa_train_ModelNet40.py fname Nstars GPUs
fname = the name of the file to save the weights to
Nstars = the number of constellation stars
GPUs = 1 if you have a single GPU

the inputs don't have defaults and it isn't robust to leaving an input blank.
"""

"""
Updated on Aug 8 to include rotation in the data augmentation
"""

import os,sys
import h5py
import numpy as np

import keras
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.optimizers import RMSprop

from myfunctions2 import MyAddScale,MyAdd3DRotation,MyAddShift,MyAddJitter,MyUrsaMin

MODEL_NET_40_DIR = os.environ['HOME']+'/'+'data/modelnet40_ply_hdf5_2048/'
#fname = sys.argv[1]
#Nstars = int(sys.argv[2])
#gpus = int(sys.argv[3])
fname = 'ModelNet40run'
Nstars=32
gpus = 1
print('fname='+fname+'_stars='+str(Nstars)+'_gpus='+str(gpus))

for j in range(5):
    infname = MODEL_NET_40_DIR+'ply_data_train'+str(j)+'.h5'
    with h5py.File(infname,'r') as f:
        if j==0:
            new_x_train = np.array(f["data"][:])
            new_y_train = keras.utils.to_categorical(f["label"][:], 40)
        else:
            new_x_train = np.concatenate((   new_x_train,   np.array(f["data"][:])   ))
            new_y_train = np.concatenate((   new_y_train,   keras.utils.to_categorical(f["label"][:], 40)   ))

for j in range(2):
    infname = MODEL_NET_40_DIR+'ply_data_test'+str(j)+'.h5'
    with h5py.File(infname,'r') as f:
        if j==0:
            new_x_test = np.array(f["data"][:])
            new_y_test = keras.utils.to_categorical(f["label"][:], 40)
        else:
            new_x_test = np.concatenate((   new_x_test,   np.array(f["data"][:])   ))
            new_y_test = np.concatenate((   new_y_test,   keras.utils.to_categorical(f["label"][:], 40)   ))


inputs = Input(shape=(new_x_train.shape[1],new_x_train.shape[2]))
#these first layers are 'data augmentation' layers
x = MyAddScale(name='scale_augment')(inputs)
x = MyAdd3DRotation(name='rotate_augment')(x)
x = MyAddShift(name='shift_augment')(x)
x = MyAddJitter(name='jitter_augment')(x)
#This is the ursa layer to create a feature vector
x = MyUrsaMin(Nstars,name='cluster')(x)
x = Activation('relu')(x)
x = BatchNormalization()(x)
#these last layers do classification
x = Dense(512,activation= 'relu',name='dense512')(x)
x = BatchNormalization()(x)
x = Dense(256,activation= 'relu',name='dense256')(x)
x = BatchNormalization()(x)
x = Dropout(rate=0.3)(x)
x = Dense(40,activation='softmax')(x)

model = Model(inputs=inputs, outputs= x)
model.summary()
if gpus>1:
    from keras.utils import multi_gpu_model
    model = multi_gpu_model(model, gpus=gpus)

rmsprop=tf.keras.optimizers.RMSprop(lr=.001, rho=.9,decay=.0001)
model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()
#model.set_weights(w)  
history = model.fit(new_x_train, 
                    new_y_train, 
                    validation_data=(new_x_test,new_y_test),
                    epochs=500, batch_size=512, verbose=2)


 
