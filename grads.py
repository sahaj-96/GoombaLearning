import tensorflow as tf
from . import consts
maximum_gradient = consts.gradient_max
num_actions = consts.num_actions
base_clip_epsilon = consts.base_clip_epsilon
value_loss_coefficient = consts.value_loss_coefficient
entropy_loss_coefficient = consts.entropy_loss_coefficient

@tf.function(experimental_relax_shapes=True)
def loss(alpha,policy_network,value_network,observation,actions,advantages,old_policies,old_values):    
    policy = policy_network(observation)    
    v = value_network(observation)
    v = tf.squeeze(v, axis = 1)    
    act_one_hot = tf.one_hot(actions, num_actions)    
    clip_epsilon = tf.math.multiply(alpha , base_clip_epsilon)
    log_prob_ratio = tf.math.log(tf.reduce_sum(policy * act_one_hot, axis=1)) - tf.math.log(tf.reduce_sum(old_policies * act_one_hot, axis=1))                                   
    prob_ratio = tf.math.exp(log_prob_ratio)             
    clipped_prob_ratio = tf.clip_by_value(prob_ratio, 1 - clip_epsilon, 1 + clip_epsilon)                                            
    entropy_loss = -tf.reduce_sum(- policy * tf.math.log(policy), axis=1)                        
    clip_loss = -tf.math.minimum(prob_ratio * advantages, clipped_prob_ratio * advantages)                                         
    value_loss = tf.math.square(v - old_values)        
    total_loss = tf.reduce_mean(clip_loss + value_loss_coefficient * value_loss + entropy_loss_coefficient * entropy_loss)            
    return entropy_loss, clip_loss, value_loss, total_loss


def gradients(optimizer):  
    @tf.function
    def apply_grads(alpha,policy_network,value_network,observation,actions,advantages,old_policies,old_values):
        
        with tf.GradientTape() as tape:                  
            entropy_loss, clip_loss, value_loss, total_loss=loss(alpha,policy_network,value_network,observation,actions,advantages,old_policies,old_values)
        variables = (policy_network.trainable_variables + value_network.trainable_variables)
        gradients = tape.gradient(total_loss, variables)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, maximum_gradient)
        optimizer.apply_gradients(zip(clipped_gradients, variables))     
        return entropy_loss, clip_loss, value_loss, total_loss
    return apply_grads