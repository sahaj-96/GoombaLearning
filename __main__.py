import argparse 
import os
from . import grads    
from . import networks
from . import environments
from . import env_maker
import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf
import random
num_actors=8
env_n="S"
maximum_steps=1e7
trajectory_len=1280
version=env_maker.version

def loader(v_nn,p_nn, path_v_nn,path_p_nn):
    if path_v_nn!=None and path_p_nn!=None:
        if(os.path.exists(path_v_nn)and os.path.exists(path_p_nn)):
            v_nn.load_weights(path_v_nn)
            p_nn.load_weights(path_p_nn)
            print(f'Loaded Value Network weights from {path_v_nn}')
            print(f'Loaded Policy Network weights from {path_p_nn}')
    else:
        print(f'Initial weights assigned to both Value and Policy Network')

def saver(v_nn,p_nn, path):
    if not os.path.exists(path):
        os.makedirs(path)
    v_nn.save_weights(os.path.join(path,("Value_Weights")))
    p_nn.save_weights(os.path.join(path,("Policy_Weights")))
    print('Saved model weight for Value and Policy Networks')

def alpha_anneal(t):
    return tf.convert_to_tensor(np.maximum(1.0 - (float(t) / float(maximum_steps)), 0.0), dtype=tf.float32)



def train(obv_shape,no_of_actions,env,path_vnn, path_pnn):
    print("Starting Model Training....")
    print("||----------------------------------------------------------||")
    print(f"Epochs                    : {options.epochs}")
    print(f"Learning Rate             : {options.learning_rate}")
    start_time=0
    actors=[]
    value_net=networks.ValueNetwork(observation_shape=obv_shape)
    policy_net=networks.PolicyNetwork(observation_shape=obv_shape,num_actions=no_of_actions)
    loader(value_net, policy_net,path_vnn, path_pnn)
    for i in range(num_actors):
        actors.append(environments.Maincontroller(environments.Envcontrol(env_n,env)))
    while i<=maximum_steps:
        upd_learning_rate = options.learning_rate *alpha_anneal(i)
        adam = tf.keras.optimizers.Adam(learning_rate=upd_learning_rate, epsilon=1e-5)   
        model_gradients = grads.gradients(adam,number_of_actions=no_of_actions)
        
        for y in range(trajectory_len):
            for z in actors:                                
                z.take_step_in_env(policy_net,value_net,i,number_of_actions=no_of_actions)
            i+=1
        for z in actors:
            z.calc_advantages(i,horizon=trajectory_len)
        obs_data = act_data = policy_data = advantage_data = value_estimate_data = []
        for z in actors:
            obs_actor,act_actor,policy_actor,advantage_actor,value_estimate_actor=z.get_data(i,trajectory_len)
            obs_data.extend(obs_actor)
            act_data.extend(act_actor)
            policy_data.extend(policy_actor)
            advantage_data.extend(advantage_actor)
            value_estimate_data.extend(value_estimate_actor)
        # Normalization
        advantage_data=np.array(advantage_data)
        advantage_data=(advantage_data-np.mean(advantage_data))/(np.std(advantage_data)+K)
        num_samples=len(obs_data)
        indices=list(range(num_samples))
        for e in range(options.epochs):
            random.shuffle(indices)
            y = 0
            while y<num_samples:
                obs_data_batch=value_data_batch=policy_data_batch=adv_data_batch=act_data_batch= []
                batch_size=128
                for b in range(batch_size):
                    index=indices[y]                    
                    obs_data_batch.append(np.squeeze(obs_data[index],axis=0))
                    act_data_batch.append(act_data[index])
                    policy_data_batch.append(policy_data[index])
                    adv_data_batch.append(advantage_data[index])
                    value_data_batch.append(value_estimate_data[index])
                    y+=1
                obs_data_batch = tf.convert_to_tensor(np.asarray(obs_data_batch), dtype=tf.float32) 
                act_data_batch = tf.convert_to_tensor(np.asarray(act_data_batch), dtype=tf.uint8) 
                policy_data_batch = tf.convert_to_tensor(np.asarray(policy_data_batch), dtype=tf.float32) 
                adv_data_batch = tf.convert_to_tensor(np.asarray(adv_data_batch), dtype=tf.float32) 
                value_data_batch = tf.convert_to_tensor(np.asarray(value_data_batch), dtype=tf.float32)               
                entropy_loss, clip_loss, value_loss, total_loss = model_gradients(
                                                                                    alpha=alpha_anneal(i),
                                                                                    p_nn=policy_net,
                                                                                    v_nn=value_net,
                                                                                    obv=obs_data_batch,
                                                                                    actions=act_data_batch,
                                                                                    adv=adv_data_batch,
                                                                                    olp_p=policy_data_batch,
                                                                                    old_v=value_data_batch,
                                                                                    number_of_actions=no_of_actions
                                                                                    )
                print(f'Entropy Loss:{entropy_loss}  Clip Loss:{clip_loss}  Value Loss:{value_loss}  Total Loss:{total_loss}')
                if options.checkpoint_dir:
                    checkpoint_file = os.path.join(options.checkpoint_dir, "checkpoint-epoch-%d" % (e))
                    saver(value_net,policy_net,checkpoint_file)
                    print("Saved model checkpoint to '%s'" % checkpoint_file)
        for z in actors:
            z.clear_history(i,trajectory_len)
        
        saver(value_net,policy_net,options.save_best_to)
        episodes_rewards = []
        episodes_x = []
        for z in actors:
            episodes_rewards.extend(z.rew_ofeach_episode)
            episodes_x.extend(z.xpos_ofeach_episode)
        if len(episodes_rewards)>=10:
            print(f"Time: {i}, AVG Reward: {np.mean(episodes_rewards):.2f}, AVG X: {np.hstack(episodes_x).mean():.2f}, MIN X: {np.hstack(episodes_x).min():.2f}, MAX X: {np.hstack(episodes_x).max():.2f}")
            for z in actors:
                z.rew_ofeach_episode = []
                z.xpos_ofeach_episode = []



if __name__ == '__main__':
    # User Interface
    parser = argparse.ArgumentParser("GoombaLearning")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--train", action = "store_true", help = "Train model")
    parser.add_argument("--save-best-to", metavar = "file", action = "store", help = "Save best weights to file")
    parser.add_argument("--checkpoint-dir", metavar = "dir", action = "store", help = "Save checkpoints after each epoch to the given directory")
    parser.add_argument("--load-value-weights-from", type = str , metavar = "file", action = "store", help = "Load initial model weights for Value Netwok from file")
    parser.add_argument("--load-policy-weights-from", type = str,metavar = "file", action = "store", help = "Load initial model weights for Policy Netwok from file")
    parser.add_argument("--epochs", metavar = "count", type = int, action = "store", default = 1, help = "Number of epochs to train for")
    parser.add_argument("--learning-rate", metavar = "value", type = float, action = "store", default = 1e-3, help = "Learning rate")
    options = parser.parse_args()

    tf.config.set_visible_devices([], 'GPU')
    cuda_available = tf.test.is_built_with_cuda()
    gpu_available = tf.config.list_physical_devices('GPU')
    if gpu_available:
        for gpu in gpu_available:
            print(gpu)
            tf.config.experimental.set_memory_growth(gpu, True)
    print("CUDA Available : %s" % ("yes" if cuda_available else "no"))
    print("GPU Available  : %s" % ("yes" if gpu_available else "no"))
    print("Eager Execution: %s" % ("yes" if tf.executing_eagerly() else "no"))

    envi,obv_shape,num_actions=env_maker.make_env(env_idx=version-1)
    if options.train:
        train(obv_shape=obv_shape,no_of_actions=num_actions,
                env=envi,path_vnn=options.load_value_weights_from,path_pnn=options.load_policy_weights_from)