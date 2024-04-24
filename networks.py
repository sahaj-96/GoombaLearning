import tensorflow as tf
import numpy as np
from . import consts
observation_shape=consts.observation_shape
num_actions=consts.num_actions
class valuenet(tf.keras.Model):
    def __init__(self):
        super(valuenet,self).__init__(name='val_weights')        
        self.convlayer1=tf.keras.layers.Conv2D(filters=32,kernel_size=(8, 8),strides=(4, 4),padding='same',input_shape=observation_shape,kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),bias_initializer=tf.keras.initializers.Zeros(),activation='linear')
        self.convlayer2=tf.keras.layers.Conv2D(filters=64,kernel_size=(4, 4),strides=(2, 2),padding='same',kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),bias_initializer=tf.keras.initializers.Zeros(),activation='linear')
        self.convlayer3=tf.keras.layers.Conv2D(filters=64,kernel_size=(3, 3),strides=(1, 1),padding='same',kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),bias_initializer=tf.keras.initializers.Zeros(),activation='linear')
        self.flatten=tf.keras.layers.Flatten()
        self.dense=tf.keras.layers.Dense(512,kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),bias_initializer=tf.keras.initializers.Zeros(),activation = 'relu')        
        self.out = tf.keras.layers.Dense(1,activation = 'linear',bias_initializer=tf.keras.initializers.Ones(),kernel_initializer=tf.keras.initializers.Orthogonal(.01))    
    @tf.function
    def call(self,inputs):
        x = self.convlayer1(inputs)        
        x = tf.keras.layers.LeakyReLU()(x)
        x = self.convlayer2(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = self.convlayer3(x)
        x = tf.keras.layers.LeakyReLU()(x)
        x = self.flatten(x)        
        x = self.dense(x)        
        return self.out(x)
    
class policynet(tf.keras.Model):
    def __init__(self):
        super(policynet,self).__init__(name='pol_weights')  
        self.convlayer1=tf.keras.layers.Conv2D(filters=32,kernel_size=(8,8),strides=(4,4),padding='same',input_shape=observation_shape,kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),bias_initializer=tf.keras.initializers.Zeros(),activation='linear')  
        self.convlayer2 =tf.keras.layers.Conv2D(filters=64,kernel_size=(4,4),strides=(2,2),padding='same',kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),bias_initializer=tf.keras.initializers.Zeros(),activation='linear')  
        self.convlayer3=tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),bias_initializer=tf.keras.initializers.Zeros(),activation='linear')   
        self.flatten=tf.keras.layers.Flatten()
        self.dense=tf.keras.layers.Dense(512,kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),bias_initializer=tf.keras.initializers.Zeros(),activation = 'relu')                
        self.out=tf.keras.layers.Dense(num_actions,bias_initializer=tf.keras.initializers.Zeros(),activation = 'softmax')
    @tf.function
    def call(self,inputs):
            x = self.convlayer1(inputs)
            x = tf.keras.layers.LeakyReLU()(x)
            x = self.convlayer2(x)
            x = tf.keras.layers.LeakyReLU()(x)
            x = self.convlayer3(x)
            x = tf.keras.layers.LeakyReLU()(x)
            x = self.flatten(x)        
            x = self.dense(x)                
            return self.out(x)