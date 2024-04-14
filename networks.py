import tensorflow as tf
import numpy as np
observation_shape =0 #TODO 
num_actions=0 #TODO

class ValueNetwork(tf.keras.Model):
    def __init__(self):
        super(ValueNetwork,self).__init__(name='Network_for_value_calc')
        
        self.convlayer1=tf.keras.layers.Conv2D(filters=32,kernel_size=(8, 8),strides=(4, 4),padding='same',input_shape=observation_shape,kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),bias_initializer=tf.keras.initializers.Zeros(),activation='linear')
        self.convlayer2=tf.keras.layers.Conv2D(filters=64,kernel_size=(4, 4),strides=(2, 2),padding='same',kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),bias_initializer=tf.keras.initializers.Zeros(),activation='linear')
        self.convlayer3=tf.keras.layers.Conv2D(filters=64,kernel_size=(3, 3),strides=(1, 1),padding='same',kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),bias_initializer=tf.keras.initializers.Zeros(),activation='linear')
        self.flatten=tf.keras.layers.Flatten()
        self.dense=tf.keras.layers.Dense(512,kernel_initializer=tf.keras.initializers.Orthogonal(np.sqrt(2)),bias_initializer=tf.keras.initializers.Zeros(),activation = 'relu')        
        self.out = tf.keras.layers.Dense(1,activation = 'linear',bias_initializer=tf.keras.initializers.Ones(),kernel_initializer=tf.keras.initializers.Orthogonal(.01))
    
    @tf.function
    def call(self,inputs):
        blocks=[self.convlayer1,self.convlayer2,self.convlayer3,self.flatten,self.dense,self.out]
        y = inputs
        for block in blocks:
            y = block(y)
        return y