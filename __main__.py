import argparse,os,numpy as np,random,cv2,tensorflow as tf
from . import environments,consts,networks,grads,env_maker

env = consts.env
obs_shape = consts.observation_shape
num_actions = consts.num_actions
start_time = consts.start_t
num_actors = consts.num_actors
maximum_steps = consts.max_steps
base_learning_rate = consts.base_learning_rate
trajectory_len=consts.horizon
batch_size=consts.batch_size
path_v_nn=".\\GoombaLearning\\temp\\value_weg"
path_p_nn=".\\GoombaLearning\\temp\\policy_weg"

def loader(v_nn,p_nn):
    v_nn.load_weights(filepath=path_v_nn)
    p_nn.load_weights(filepath=path_p_nn)
    print(f'Loaded Value and Policy Network weights')


def save_weights(v_nn,p_nn):
    v_nn.save_weights(filepath = path_v_nn,overwrite=True)
    print(f"Saved Value Network model weights.")
    p_nn.save_weights(filepath = path_p_nn,overwrite=True)
    print(f"Saved Policy network model weights.")


def alpha_anneal(t):
    return tf.convert_to_tensor(np.maximum(1.0 - (float(t) / float(maximum_steps)), 0.0), dtype=tf.float32)

def train():
    print("Starting Model Training....")
    print("||----------------------------------------------------------||")
    print(f"Epochs                    : {options.epochs}")
    print(f"Learning Rate             : {options.learning_rate}")
    k=start_time
    last_save=0
    actors=[]
    value_net=networks.valuenet()
    policy_net=networks.policynet()
    loader(v_nn=value_net,p_nn= policy_net)
    for z in range(num_actors):
        actors.append(environments.EnvActor(environments.Envcontrol("s")))
    while k<=maximum_steps:
        learning_rate = base_learning_rate * alpha_anneal(k)        
        adam = tf.keras.optimizers.Adam(learning_rate=learning_rate, epsilon=1e-5)   
        g = grads.gradients(adam)
        for ii in range(trajectory_len):
            for actor in actors:                                
                actor.step_env(policy_net,value_net,t=k,num_actions=num_actions)
            k+=1
        for z in actors:
            z.calculate_horizon_advantages(k)
        obs_data =[]
        act_data =[] 
        policy_data =[] 
        adv_est_horizon =[]
        value_estimate_data = []
    
        for z in actors:
            obs_actor,act_actor,policy_actor,advantage_actor,value_estimate_actor = z.get_horizon(k)
            obs_data.extend(obs_actor)
            act_data.extend(act_actor)
            policy_data.extend(policy_actor)
            adv_est_horizon.extend(advantage_actor)
            value_estimate_data.extend(value_estimate_actor)
        adv_est_horizon = np.array(adv_est_horizon)
        adv_est_horizon = (adv_est_horizon - np.mean(adv_est_horizon)) / (np.std(adv_est_horizon) + 1e-8)
        num_samples = len(obs_data)
        indices = list(range(num_samples))

        for e in range(1,1+options.epochs):
            print(f"Epoch       {e}/{options.epochs}")
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
                #print(obs_batch)
                #Calling Function of training                
                entropy_loss, clip_loss, value_loss, total_loss = g(alpha_anneal(k),
                                                                    policy_net, value_net,obs_batch,
                                                                    act_batch,adv_batch,policy_batch,value_sample_batch)
                
                #print(f"Entropy Loss:{entropy_loss}  Clip Loss:{clip_loss}  Value Loss:{value_loss}  Total Loss:{total_loss}")

        for actor in actors:
            actor.flush(k)

        if k-last_save>1000:
            save_weights(v_nn=value_net,p_nn=policy_net)
            last_save=k

        all_ep_rewards = []
        all_ep_x = []
        for actor in actors:
            all_ep_rewards.extend(actor.episode_rewards)
            all_ep_x.extend(actor.episode_x)
        if len(all_ep_rewards) >= 10:
            print("T: %d" % (k,))
            print("AVG Reward: %f" % (np.mean(all_ep_rewards),))
            print("MIN Reward: %f" % (np.amin(all_ep_rewards),))
            print("MAX Reward: %f" % (np.amax(all_ep_rewards),))            
            print("AVG X: %f" % (np.hstack(all_ep_x).mean(),))
            print("MIN X: %f" % (np.hstack(all_ep_x).min(),))
            print("MAX X: %f" % (np.hstack(all_ep_x).max(),))
            for actor in actors:
                actor.episode_rewards = []
                actor.episode_x = []
        
        episodes_rewards = []
        episodes_x = []
        for z in actors:
            episodes_rewards.extend(z.episode_rewards)
            episodes_x.extend(z.episode_x)
        if len(episodes_rewards)>=5:
            for z in actors:
                z.episode_rewards = []
                z.episode_x = []


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
        train()
        
    if options.test:
        test()