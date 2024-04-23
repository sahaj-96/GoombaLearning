from multiprocessing import Pipe,Process
import tensorflow as tf,numpy as np
from . import consts
starting_time=consts.start_t
number_of_actions=consts.num_actions 
gamma=consts.gamma
horizon=consts.horizon
gae_lambda=consts.gae_lambda
env=consts.env
class Envcontrol:
    def controller(name,connection):
        while True:
            (command,args,kwargs)=connection.recv()
            if command=="reset":
                connection.send(env.reset())
            elif command=="step":
                connection.send(env.step(*args,**kwargs))
            elif command=="exit":
                break
            else:
                raise Exception(f"Unknown command {command}")
    def __init__(self,n):
        parent_side,child_side=Pipe()
        self.process=Process(target=Envcontrol.controller,args=(n,child_side))
        self.process.start()
        self.connection=parent_side
    def reset(self):
        self.connection.send(("reset",(),{}))
        return self.connection.recv()
    def step(self, action):
        self.connection.send(("step",(action,),{}))
        return self.connection.recv()
    def exit(self):
        self.connection.send(("exit",(),{}))
        self.process.join()


class TimeIndexedList(object):
    def __init__(self, first_t=0):
        self.first_t = first_t
        self.list = []
    def flush_through(self, t):
        to_remove = t - self.first_t + 1
        if to_remove > 0:
            self.list = self.list[to_remove:]
            self.first_t = t + 1

    def append(self, elem):     
        self.list.append(elem)

    def get(self, t):
        return self.list[t - self.first_t]

    def future_length(self):
        return len(self.list)

    def get_range(self, t, length):
        return self.list[(t - self.first_t):(t - self.first_t + length)]

class EnvActor(object):
    def __init__(self, env):
        self.env = env
        self.obs = TimeIndexedList(first_t = starting_time)
        self.last_obs = self.env.reset()
        self.last_obs = np.expand_dims(self.last_obs , axis = 0)     
        self.last_obs = tf.convert_to_tensor(self.last_obs, dtype=tf.float32)     # for performance
        self.obs.append(self.last_obs)
        # self.pobs = TimeIndexedList(first_t = start_t)
        # self.last_pobs = preprocess_obs_atari(self.obs, self.pobs, start_t, start_t)
        # self.pobs.append(self.last_pobs)
        self.act = TimeIndexedList(first_t = starting_time)
        self.rew = TimeIndexedList(first_t = starting_time)
        self.val = TimeIndexedList(first_t = starting_time)
        self.policy = TimeIndexedList(first_t = starting_time)
        self.delta = TimeIndexedList(first_t = starting_time)
        self.done = TimeIndexedList(first_t = starting_time)
        self.episode_start_t = 0
        self.episode_rewards = []
        self.episode_x = []
        self.rewards_this_episode = []
        self.x_this_episode = []
        self.advantage_estimates = TimeIndexedList(first_t = starting_time)
        self.value_estimates = TimeIndexedList(first_t = starting_time)

    def step_env(self,policy_net,value_net,t,num_actions):      
        if t == starting_time:
            # Artifact of ordering             
            val_0 = value_net(self.last_obs).numpy()[0]
            self.val.append(val_0[0])
        
        policy_t = policy_net(self.last_obs).numpy()[0]                
        # print(policy_t)
        action_t = np.random.choice(num_actions, 1, p=policy_t)[0]        

        
        obs_tp1,rew_t,done_t,info_t = self.env.step(action_t)        
        obs_tp1 = np.expand_dims(obs_tp1,axis = 0)

        self.act.append(action_t)
        self.rew.append(rew_t)
        self.policy.append(policy_t)
        self.rewards_this_episode.append(rew_t)        
        self.x_this_episode.append(info_t.get('x_pos')) 

        if done_t:
            self.done.append(True)
            self.episode_rewards.append(sum(self.rewards_this_episode))
            self.rewards_this_episode = []
            self.episode_x.append(self.x_this_episode)
            self.x_this_episode = []
            obs_tp1 = self.env.reset()
            obs_tp1 = np.expand_dims(obs_tp1,axis = 0)
            self.episode_start_t = t + 1
        else:
            self.done.append(False)
        self.obs.append(obs_tp1)
        obs_tp1 = tf.convert_to_tensor(obs_tp1, dtype=tf.float32)     # for performance 
        val_tp1 = value_net(obs_tp1).numpy()[0]       
        self.val.append(val_tp1[0])
        print(self.rew.get(t))
        print( self.val.get(t + 1))
        print( self.val.get(t))  
        if done_t:
            self.delta.append(self.rew.get(t) - self.val.get(t))       
        else:
            self.delta.append(self.rew.get(t) + gamma * self.val.get(t + 1) - self.val.get(t))
        self.last_obs = obs_tp1        

    def calculate_horizon_advantages(self, end_t):
        advantage_estimates = []
        value_estimates = []
        advantage_so_far = 0
        last_value_sample = self.val.get(end_t)
        for ii in range(horizon):
            if self.done.get(end_t - ii - 1):
                advantage_so_far = 0
                last_value_sample = 0
            advantage_so_far = self.delta.get(end_t - ii - 1) + (gamma * gae_lambda * advantage_so_far)
            advantage_estimates.append(advantage_so_far)
            last_value_sample = gamma * last_value_sample + self.rew.get(end_t - ii - 1)
            value_estimates.append(last_value_sample)
        advantage_estimates.reverse()
        value_estimates.reverse()
        for ii in range(len(advantage_estimates)):
            self.advantage_estimates.append(advantage_estimates[ii])
            self.value_estimates.append(value_estimates[ii])


    def get_horizon(self, end_t):
        return (self.obs.get_range(end_t - horizon, horizon),
                self.act.get_range(end_t - horizon, horizon),
                self.policy.get_range(end_t - horizon, horizon),
                self.advantage_estimates.get_range(end_t - horizon, horizon),
                self.value_estimates.get_range(end_t - horizon, horizon))

    def flush(self, end_t):
        self.obs.flush_through(end_t - horizon - 5)
        self.act.flush_through(end_t - horizon - 1)
        self.rew.flush_through(end_t - horizon - 1)
        self.val.flush_through(end_t - horizon - 1)
        self.policy.flush_through(end_t - horizon - 1)
        self.delta.flush_through(end_t - horizon - 1)
        self.done.flush_through(end_t - horizon - 1)
