import tensorflow as tf
from . import consts
maximum_gradient = consts.gradient_max
number_of_actions = consts.num_actions
base_clip_epsilon = consts.base_clip_epsilon
value_loss_coefficient = consts.value_loss_coefficient
entropy_loss_coefficient = consts.entropy_loss_coefficient

@tf.function
def loss(old_p,old_v,p_nn,v_nn,obv,actions,adv,alpha):    
    policy=p_nn(obv)    
    v=v_nn(obv)#compute values
    v=tf.squeeze(v,axis=1)    
    act_one_hot=tf.one_hot(actions,number_of_actions)
    clip_epsilon=tf.math.multiply(alpha,base_clip_epsilon)
    log_prob_r=tf.math.log(tf.reduce_sum(policy*act_one_hot, axis=1))-tf.math.log(tf.reduce_sum(old_p*act_one_hot, axis=1))                                   
    prob_r=tf.math.exp(log_prob_r)  
    clipped_prob_r=tf.clip_by_value(prob_r,1-clip_epsilon,1+clip_epsilon)
    entropy_loss=-tf.reduce_sum(-policy*tf.math.log(policy),axis=1)   
    clip_loss=-tf.math.minimum(prob_r*adv,clipped_prob_r*adv) 
    value_loss=tf.math.square(v-old_v)        
    total_loss=tf.reduce_mean(clip_loss+value_loss_coefficient*value_loss+entropy_loss_coefficient*entropy_loss)
    return entropy_loss,clip_loss,value_loss,total_loss

def gradients(optimizer):  
    @tf.function      #Dynamic Polymorphism
    def grads(old_p,old_v,p_nn,v_nn,obv,actions,adv,alpha):
        with tf.GradientTape() as tape:                  
            entropy_loss,clip_loss,value_loss,total_loss=loss(old_p,old_v,p_nn,v_nn,obv,actions,adv,alpha)
        all_variables=(p_nn.trainable_variables+v_nn.trainable_variables)
        initial_grads=tape.gradient(total_loss,all_variables)
        clipped_grads,_=tf.clip_by_global_norm(initial_grads,maximum_gradient)
        optimizer.apply_gradients(zip(clipped_grads,all_variables))     
        return entropy_loss,clip_loss,value_loss,total_loss
    return grads
