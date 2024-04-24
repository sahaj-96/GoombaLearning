from __future__ import absolute_import, division, print_function, unicode_literals

import argparse,os,numpy as np,random,cv2,tensorflow as tf
from . import environments,consts,networks,grads,env_maker

env =consts.env
obs_shape = consts.observation_shape
num_actions = consts.num_actions
start_time = consts.start_t
num_actors = consts.num_actors
maximum_steps = consts.max_steps
base_learning_rate = consts.base_learning_rate
trajectory_len=consts.horizon
batch_size=consts.batch_size
SMALL_NUM=1e-8
def loader(v_nn,p_nn, path_v_nn,path_p_nn):
    if path_v_nn!=None and path_p_nn!=None:
        if(os.path.exists(path_v_nn)and os.path.exists(path_p_nn)):
            v_nn.load_weights(path_v_nn)
            p_nn.load_weights(path_p_nn)
            print(f'Loaded Value Network weights from {path_v_nn}')
            print(f'Loaded Policy Network weights from {path_p_nn}')
    else:
        print(f'Initial weights assigned to both Value and Policy Network')

class BestWeightsTracker:
    def __init__(self, fpnn,fvnn):
        self._fpnn = fpnn
        self._fvnn = fvnn
        self._best_weights_vnn = None
        self._best_weights_pnn = None
        self._best_total_loss = 1000000000000

    def on_epoch_end(self, v_nn,p_nn, loss):
        if loss< self._best_total_loss:
            self._best_total_loss = loss
            self._best_weights_vnn = v_nn.get_weights()
            self._best_weights_pnn = p_nn.get_weights()

    def save_best_weights(self, v_nn,p_nn):
        if self._best_weights_vnn is not None and self._best_weights_pnn is not None:
            v_nn.set_weights(self._best_weights_vnn)
            v_nn.save_weights(filepath = self._fpnn, overwrite = True)
            print(f"Saved best Value Network model weights.")
            p_nn.set_weights(self._best_weights_pnn)
            p_nn.save_weights(filepath = self._fvnn, overwrite = True)
            print(f"Saved best Policy network model weights.")
        else:
            print("No weights have been saved yet.")


def alpha_anneal(t):
    return tf.convert_to_tensor(np.maximum(1.0 - (float(t) / float(maximum_steps)), 0.0), dtype=tf.float32)


def train(path_vnn, path_pnn):
    print("Starting Model Training....")
    print("||----------------------------------------------------------||")
    print(f"Epochs                    : {options.epochs}")
    print(f"Learning Rate             : {options.learning_rate}")
    t=start_time
    actors=[]
    value_net=networks.valuenet()
    policy_net=networks.policynet()
    loader(value_net, policy_net,path_vnn, path_pnn)
    for ii in range(num_actors):
        actors.append(environments.EnvActor(environments.Envcontrol("s")))
    while t<=maximum_steps:
        upd_learning_rate = options.learning_rate *alpha_anneal(t)
        adam = tf.keras.optimizers.Adam(learning_rate=upd_learning_rate, epsilon=1e-5)   
        model_gradients = grads.gradients(adam)
        
        for ii in range(trajectory_len):
            for z in actors:                                
                z.step_env(policy_net,value_net,t,num_actions)
            t+=1
        for z in actors:
            z.calculate_horizon_advantages(t)
        obs_data =[]
        act_data =[] 
        policy_data =[] 
        adv_est_horizon =[]
        value_estimate_data = []
        for z in actors:
            obs_actor,act_actor,policy_actor,advantage_actor,value_estimate_actor=z.get_horizon(t)
            obs_data.extend(obs_actor)
            act_data.extend(act_actor)
            policy_data.extend(policy_actor)
            adv_est_horizon.extend(advantage_actor)
            value_estimate_data.extend(value_estimate_actor)
        num_samples=len(obs_data)
        indices=list(range(num_samples))
        adv_est_horizon = np.array(adv_est_horizon)
        adv_est_horizon = (adv_est_horizon - np.mean(adv_est_horizon)) / (np.std(adv_est_horizon) + SMALL_NUM)
        for e in range(1,1+options.epochs):
            print(f"Epoch       {e}/{options.epochs}")
            bwt=BestWeightsTracker(fpnn = options.save_best_to_pnn,fvnn = options.save_best_to_vnn)
            random.shuffle(indices)
            ii = 0
            while ii < num_samples:
                obs_batch = []
                act_batch = []
                policy_batch = []
                adv_batch = []
                value_sample_batch = []

                for b in range(batch_size):
                    index = indices[ii]
                    obs_batch.append(np.squeeze(obs_data[index],axis=0))
                    act_batch.append(act_data[index])
                    policy_batch.append(policy_data[index])
                    adv_batch.append(adv_est_horizon[index])
                    value_sample_batch.append(value_estimate_data[index])
                    ii += 1

                # Training loop
                obs_batch = tf.convert_to_tensor(np.asarray(obs_batch), dtype=tf.float32) 
                act_batch = tf.convert_to_tensor(np.asarray(act_batch), dtype=tf.uint8) 
                policy_batch = tf.convert_to_tensor(np.asarray(policy_batch), dtype=tf.float32) 
                adv_batch = tf.convert_to_tensor(np.asarray(adv_batch), dtype=tf.float32) 
                value_sample_batch = tf.convert_to_tensor(np.asarray(value_sample_batch), dtype=tf.float32)                                  
                entropy_loss, clip_loss, value_loss, total_loss = model_gradients(
                                                                                    alpha_anneal(t),
                                                                                    policy_net,
                                                                                    value_net,
                                                                                    obs_batch,
                                                                                    act_batch,
                                                                                    adv_batch,
                                                                                    policy_batch,
                                                                                    value_sample_batch
                                                                                    )
                print(f"Entropy Loss:{entropy_loss}") 
                print(f"Clip Loss:{clip_loss} ")
                print(f"Value Loss:{value_loss} ")
                print(f"Total Loss:{total_loss} ")
                if options.save_best_to_pnn and options.save_best_to_vnn:
                    bwt.on_epoch_end(value_net,policy_net,total_loss)
        if options.save_best_to_pnn and options.save_best_to_vnn:
            bwt.save_best_weights(value_net,policy_net)
        for z in actors:
            z.flush(t)
        
        episodes_rewards = []
        episodes_x = []
        for z in actors:
            episodes_rewards.extend(z.episode_rewards)
            episodes_x.extend(z.episode_x)
        if len(episodes_rewards)>=5:
            print(f"Time: {t}, AVG Reward: {np.mean(episodes_rewards):.2f}, AVG X: {np.hstack(episodes_x).mean():.2f}, MIN X: {np.hstack(episodes_x).min():.2f}, MAX X: {np.hstack(episodes_x).max():.2f}")
            for z in actors:
                z.episode_rewards = []
                z.episode_x = []
    return


def test(path_p_nn,episodes):
    env_test=env_maker.make_env(1)
    policy_n=networks.policynet()
    policy_n.load_weights(path_p_nn)
    print(f"Loaded Model")
    done = False
    scores = []
    for e in range(episodes):
        state=env_test.reset()
        state=np.expand_dims(state , axis = 0)     
        state=tf.convert_to_tensor(state, dtype=tf.float32)      
        score=0
        video_frames = []
        while True:
            video_frames.append(cv2.cvtColor(env_test.render(mode = 'rgb_array'), cv2.COLOR_RGB2BGR))
            p = policy_n(state).numpy()[0]                            
            act=np.argmax(p) # Deterministic action            
            state,reward,done,_=env_test.step(act)        
            state=np.expand_dims(state,axis = 0)
            state=tf.convert_to_tensor(state, dtype=tf.float32)     
            score+=reward 
            if done:
                break
                
        save_name= f"supermario_rl_bot_v{consts.version}_ep{e}.mp4"
        _, height, width,_=np.shape(video_frames)
        vw=cv2.VideoWriter_fourcc(*'mp4v')
        video=cv2.VideoWriter(save_name,vw, 5, (width,height))
        for image in video_frames:            
            video.write(image)
        cv2.destroyAllWindows()
        video.release()        
        print('Test #%s , Score: %0.1f' %(e, score))    
        scores.append(score)
    print('Average reward: %0.2f of %s episodes' %(np.mean(scores),episodes))  
    return 

if __name__ == '__main__':
    # User Interface
    parser = argparse.ArgumentParser("GoombaLearning")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--train", action = "store_true", help = "Train model")
    group.add_argument("--test", action = "store_true", help = "Test model")
    parser.add_argument("--save-best-to-pnn", metavar = "file", action = "store", help = "Save best weights to pnn file")
    parser.add_argument("--save-best-to-vnn", metavar = "file", action = "store", help = "Save best weights to vnn file")
    parser.add_argument("--load-value-weights-from", type = str , default = None,metavar = "file", action = "store", help = "Load initial model weights for Value Netwok from file")
    parser.add_argument("--load-policy-weights-from", type = str,default = None,metavar = "file", action = "store", help = "Load initial model weights for Policy Netwok from file")
    parser.add_argument("--epochs", metavar = "count", type = int, action = "store", default = 1, help = "Number of epochs to train for")
    parser.add_argument("--episodes", metavar = "count", type = int, action = "store", default = 1, help = "Number of episodes to test for")
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

    if options.train:
        train(path_vnn=options.load_value_weights_from,path_pnn=options.load_policy_weights_from)
        
    if options.test:
        test(path_p_nn=options.load_policy_weights_from,episodes=options.episodes)