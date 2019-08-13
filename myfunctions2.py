# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 17:28:26 2018
Copyright (c) 2018 Mark B. Skouson

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from __future__ import print_function

import tensorflow as tf
import tensorflow.keras.backend as K


from tensorflow.keras.initializers import RandomUniform


import numpy as np
#import matplotlib.pyplot as plt

import random


class MyAdd3DRotation(tf.layers.Layer):
    def __init__(self, upper =.087, lower=None, **kwargs): #.087 is about 5 degrees
        #upper and lower are in radians.  If you only specify 1, then it will be plus or minus that rotation amount
        if lower ==None:
            lower = -upper
        self.lower = lower
        self.upper = upper
        super(MyAdd3DRotation, self).__init__(**kwargs)
    def build(self, input_shape):
        self.inshape = input_shape
        super(MyAdd3DRotation, self).build(input_shape)
    def call(self, x, training=None):
        def noised():
            Ni = tf.shape(x)[0] #This is the number in the batch
            #get an angle to shift each image in the batch
            """This is going to get a uniformly distributed random 3d vector 
            of length 1.  This will  be the axis of rotation.
            Then it will pick a random rotation amount around that axis. Amount 
            of rotation is based on the input to the class.
            It will use quaternion math to convert that axis and rotation to a 
            rotation matrix.  
            The rotation matrix will be the transpose of what is in the 
            literature so we can do a post multiply rather than a pre multiply
            so we don't have to change x around.
            """
            vz = tf.random_uniform((Ni,),  -1.0 , 1.0 )
            theta = tf.random_uniform((Ni,), 0.0, 2*np.pi)
            r = tf.sqrt(1-vz*vz)
            vx = r * tf.cos(theta)
            vy = r * tf.sin(theta)
            # So (vx,vy,vz) is a point on the unit sphere with
            # equal prob of being anywhere.
            # This will be the axis of rotation

            #Now pick the amount of the rotation
            ang = tf.random_uniform((Ni,), self.lower, self.upper)
            #apply this rotation axis selected above to create the quaternion
            # the result is a quaternion (qw,qx,qy,qz)
            #https://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToQuaternion/index.htm
            qw = tf.cos(ang/2)
            sinangd2 = tf.sin(ang/2)
            qx=vx*sinangd2
            qy = vy*sinangd2
            qz = vz*sinangd2

            #so we have a quaternion that has (qw,qx,qy,qz)
            #turn that into a rotation matrix:
            #http://www.neil.dantam.name/note/dantam-quaternion.pdf
            #and
            #http://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/
            #precompute some values
            txx = 2*qx*qx
            tyy = 2*qy*qy
            tzz = 2*qz*qz
            txy = 2*qx*qy
            tyz = 2*qy*qz
            txz = 2*qx*qz
            txw = 2*qx*qw
            tyw = 2*qy*qw
            tzw = 2*qz*qw
            
            #this is the transpose of what is in the literature.
            R = tf.stack([
                    tf.stack([1-tyy-tzz, txy-tzw, txz+tyw],axis=-1),
                    tf.stack([txy+tzw,1-txx-tzz,tyz-txw],axis=-1),
                    tf.stack([txz-tyw,tyz+txw,1-txx-tyy],axis=-1)],
                    axis=-1)
            #remember because of the shape of x, must pust multiply with R    
            return tf.matmul(x,R) 
        return K.in_train_phase(noised, x, training=training)
    def compute_output_shape(self, input_shape):
        return input_shape

class MyAdd2DRotation(tf.layers.Layer):
    def __init__(self, amount = .06, lower=-0.18, upper=.18, **kwargs):
        self.amount = amount
        self.lower = lower
        self.upper = upper
        super(MyAdd2DRotation, self).__init__(**kwargs)
    def build(self, input_shape):
        self.inshape = input_shape
        super(MyAdd2DRotation, self).build(input_shape)
    def call(self, x, training=None):
        def noised():
            Ni = K.shape(x)[0] #This is the number in the batch
            #get an angle to shift each image in the batch
            angles = K.clip( self.amount*K.random_normal((Ni,)),   self.lower,   self.upper)
            #We are going to post multiply the vector (x'=xR) with the matrix 
            #rather than the normal way (x'=Rx)
            #so we use the transpose of what is shown in literature for R
            R = K.stack( (K.stack((K.cos(angles),K.sin(angles)),axis=1)  ,
                           K.stack((-K.sin(angles),K.cos(angles)),axis=1))  ,
                         axis=1)
            return tf.matmul(x,R) 
        return K.in_train_phase(noised, x, training=training)
    def compute_output_shape(self, input_shape):
        return input_shape


class MyAddJitter(tf.layers.Layer):
    def __init__(self, amount=0.01, lower=-.05, upper=.05, **kwargs):
        self.amount = amount
        self.lower = lower
        self.upper = upper
        super(MyAddJitter, self).__init__(**kwargs)
    def build(self, input_shape):
        self.inshape = input_shape
        super(MyAddJitter, self).build(input_shape)
    def call(self, x, training=None):
        def noised():
            return x + K.clip(  self.amount*K.random_normal(K.shape(x)),   self.lower,   self.upper)
        return K.in_train_phase(noised, x, training=training)
    def compute_output_shape(self, input_shape):
        return input_shape

class MyAddScale(tf.layers.Layer):
    def __init__(self, lower=0.8, upper=1.25, **kwargs):
        self.lower = lower
        self.upper = upper
        super(MyAddScale, self).__init__(**kwargs)
    def build(self, input_shape):
        self.inshape = input_shape
        super(MyAddScale, self).build(input_shape)
    def call(self, x, training=None):
        def noised():
            Ni = tf.shape(x)[0] #This is the number in the batch
            #Now generate a random number for each batch image and multiply each
            # point in each bacth image by that number
            return x * tf.random_uniform((Ni,), self.lower,self.upper)[:,None,None]
        return K.in_train_phase(noised, x, training=training)
    def compute_output_shape(self, input_shape):
        return input_shape

class MyAddShift(tf.layers.Layer):
    def __init__(self, lower=-0.1, upper=0.1, **kwargs):
        self.lower = lower
        self.upper = upper
        super(MyAddShift, self).__init__(**kwargs)
    def build(self, input_shape):
        self.inshape = input_shape
        super(MyAddShift, self).build(input_shape)
    def call(self, x, training=None):
        def noised():
            Ni = K.shape(x)[0]
            Nipd = K.shape(x)[2]
            return x + K.random_uniform((Ni,Nipd), self.lower,self.upper)[:,None,:]
        return K.in_train_phase(noised, x, training=training)
    def compute_output_shape(self, input_shape):
        return input_shape

class MyUrsaMin(tf.layers.Layer):  #this initializes between -1 and 1 
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyUrsaMin, self).__init__(**kwargs)
    def build(self, input_shape):
        self.stars = self.add_weight(name = 'stars',
                                      shape = (self.output_dim, input_shape[2]),
                                      initializer = RandomUniform(minval=-1,maxval=1),
                                      trainable = True)
        super(MyUrsaMin, self).build(input_shape)
    def call(self, x):
        #shape [Batchsize,Nstars,Npoints,Ndimensions]
        diff = x[:,None,:,:]-self.stars[None,:,None,:] #difference between each input point and each star in the volume
        dists = K.min ( 
                        tf.norm(diff, axis=3),  #euclidean distance between each input point and each star
                        axis = 2) ## For each star, find the distance to the closest input point 
        return dists
    def compute_output_shape(self, input_shape):
        return(input_shape[0], self.output_dim)

''' This is not yet working
class MyUrsaSegMin(tf.layers.Layer):  #this initializes between -1 and 1 
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyUrsaSegMin, self).__init__(**kwargs)
    def build(self, input_shape):
        self.stars = self.add_weight(name = 'stars',
                                      shape = (self.output_dim, input_shape[2]),
                                      initializer = RandomUniform(minval=-1,maxval=1),
                                      trainable = True)
        super(MyUrsaSegMin, self).build(input_shape)
    def call(self, x):
        #Shape may be a little different from the layers used for classification
        #shape [Batchsize,Npoints,Nstars,Ndimensions]
        diff = x[:,:,None,:]-self.stars[None,None,:,:] #difference between each input point and each star in the volume
        pointfeatures = tf.norm(diff, axis=3)   #euclidean distance between each input point and each star
        globalfeatures = K.min(pointfeatures, axis = 1) ## For each star, find the distance to the closest input point 
        return tf.concat([x,tf.tile(globalfeatures[:,None,:],(1,x.shape[1],1))],axis=2)
    def compute_output_shape(self, input_shape):
        return(input_shape[0],input_shape[1], self.output_dim*2)
'''

class MyUrsaExp(tf.layers.Layer):  
    def __init__(self, output_dim, sigma=10, **kwargs):
        self.output_dim = output_dim
        self.sigma = sigma
        super(MyUrsaExp, self).__init__(**kwargs)
    def build(self, input_shape):
        self.stars = self.add_weight(name = 'stars',
                                      shape = (self.output_dim, input_shape[2]),
                                      initializer = RandomUniform(minval=-1,maxval=1),
                                      trainable = True)
        super(MyUrsaExp, self).build(input_shape)
    def call(self, x):
        diff = x[:,None,None,:,:]-self.stars[None,:,:,None,:] #difference between each input point and each star in the volume
        #size could be [60000, 256,1,312,2]
        dists = K.sum ( 
                        K.exp( -self.sigma*tf.norm(diff, axis=3)),  
                        axis = 2) #
        return dists
    def compute_output_shape(self, input_shape):
        return(input_shape[0], self.output_dim)

class MyUrsaGau(tf.layers.Layer):
    def __init__(self, output_dim, sigma=0.1, **kwargs):
        self.output_dim = output_dim
        self.sigma = sigma     
        self.s = 1/(2*sigma*sigma) #default = 50
        super(MyUrsaGau, self).__init__(**kwargs)
    def build(self, input_shape):
        self.stars = self.add_weight(name = 'stars',
                                      shape = (self.output_dim,input_shape[2]),
                                      initializer = RandomUniform(minval=-1,maxval=1),
                                      trainable = True)
        super(MyUrsaGau, self).build(input_shape)
    def call(self, x):
        diff = x[:,None,:,:]-self.stars[None,:,None,:] #difference between each input point and each star in the volume
        dists = K.sum (
                        K.exp( -self.s*K.sum(diff*diff, axis=3)),  #Gaussian RBF kernel
                        axis = 2) #
        return dists
    def compute_output_shape(self, input_shape):
        return(input_shape[0], self.output_dim)


def find_max_points(indata):
    maxpoints=0
    minpoints=1000
    for image in indata:  #pull an image from the set
        tmp1 = np.argwhere(image>128)  #find coords for pixels >128
        maxpoints = max(maxpoints,tmp1.shape[0])
        minpoints = min(minpoints,tmp1.shape[0])
    return maxpoints

def turn_to_point_centric(indata,maxpoints):
    #Find how much memory to allocate
    dims = len(indata.shape)-1 #dimensions of the points
    nimages=0
    for image in indata:  #pull an image from the set
        nimages+=1
    
    #allocate the memory for the new data set    
    outdata_x = np.empty([nimages,maxpoints,dims],dtype='float32')
    
    #transform the data to point centric
    for i,image in enumerate(indata):  #pull an image from the set
        tmp1 = np.argwhere(image>128)  #find coords for pixels >128
        npoints = tmp1.shape[0]
        tmp2= np.random.randint(npoints,size=maxpoints)
        tmp2[0:npoints]=np.arange(npoints)
        tmp3 = tmp1[tmp2,:]
        outdata_x[i,:,:] = tmp3
    return outdata_x

