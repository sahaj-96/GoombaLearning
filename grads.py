import tensorflow as tf
base_clip_epsilon = 0.2
value_loss_coefficient = .01
entropy_loss_coefficient = .01
maximum_gradient = 10.0
@tf.function
def loss(old_p,old_v,p_nn,v_nn,obv,actions,adv,alpha,number_of_actions):    
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

def gradients(optimizer,number_of_actions):  
    @tf.function      #Dynamic Polymorphism
    def grads(old_p,old_v,p_nn,v_nn,obv,actions,adv,alpha,number_of_actions):
        with tf.GradientTape() as tape:                  
            entropy_loss,clip_loss,value_loss,total_loss=loss(old_p,old_v,p_nn,v_nn,obv,actions,adv,alpha,number_of_actions)
        all_variables=(p_nn.trainable_variables+v_nn.trainable_variables)
        initial_grads=tape.gradient(total_loss,all_variables)
        clipped_grads,_=tf.clip_by_global_norm(initial_grads,maximum_gradient)
        optimizer.apply_gradients(zip(clipped_grads,all_variables))     
        return entropy_loss,clip_loss,value_loss,total_loss
    return grads
