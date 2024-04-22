from multiprocessing import Pipe,Process
import tensorflow as tf
import numpy as np
starting_time=0
gamma=.99
gae_lambda=0.95
from . import env_maker
env=env_maker.envi
class Envcontrol(object):
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
    def __init__(self,name,env):
        parent_side,child_side=Pipe()
        self.process=Process(target=Envcontrol.controller,args=(name,child_side))
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


class TimeOrderedList:
    def __init__(self,first_time=0):
        self.first_time=first_time
        self.list=[]
    def remove_excess(self,t):
        for_removing=t-self.first_time+1
        if for_removing>0:
            self.list=self.list[for_removing:]
            self.first_time=t+1
    def append(self,to_add):
        self.list.append(to_add)
    def get(self, t):
        return self.list[t - self.first_time]
    def future_len(self):
        return len(self.list)
    def in_range(self,t,length):
        return self.list[(t-self.first_time):(t-self.first_time+length)]


class Maincontroller:
    def __init__(self, environment):
        self.env=environment
        self.observation=TimeOrderedList(first_time=starting_time) 
        self.last_observation = self.env.reset()
        self.last_observation = np.expand_dims(self.last_observation , axis = 0)     
        self.last_observation = tf.convert_to_tensor(self.last_observation, dtype=tf.float32) 
        self.observation.append(self.last_observation)
        self.act = TimeOrderedList(first_time=starting_time)
        self.rew =TimeOrderedList(first_time=starting_time)
        self.val = TimeOrderedList(first_time=starting_time)
        self.policy =TimeOrderedList(first_time=starting_time)
        self.delta =TimeOrderedList(first_time=starting_time) 
        self.done = TimeOrderedList(first_time=starting_time)
        self.episode_start_time = 0
        self.rew_ofeach_episode = []
        self.xpos_ofeach_episode = []
        self.rew_ofcurr_episode =[]
        self.xpos_ofcurr_episode = []
        self.estimate_the_advantages =TimeOrderedList(first_time=starting_time)
        self.estimate_the_values = TimeOrderedList(first_time=starting_time)

    def take_step_in_env(self,p_nn,v_nn,t,number_of_actions):      
        if t==starting_time:
            val_0=v_nn(self.last_observation).numpy()[0]
            self.val.append(val_0[0])
        policy_t=p_nn(self.last_observation).numpy()[0]
        action_t=np.random.choice(number_of_actions,1,p=policy_t)[0] #1 because i need to draw a single action
        next_state,step_reward,episode_done,step_info=self.env.step(action_t)        
        next_state=np.expand_dims(next_state,axis=0)
        self.act.append(action_t)
        self.rew.append(step_reward)
        self.policy.append(policy_t)
        self.rew_ofcurr_episode.append(step_reward)        
        self.xpos_ofcurr_episode.append(step_info.get('x_pos'))
        if episode_done:
            self.done.append(True)
            self.rew_ofeach_episode.append(sum(self.rew_ofcurr_episode))
            self.rew_ofcurr_episode=[]
            self.xpos_ofeach_episode.append(self.xpos_ofcurr_episode)
            self.xpos_ofcurr_episode=[]
            next_state=self.env.reset()
            next_state=np.expand_dims(next_state,axis=0)
            self.episode_start_time=t+1
        else:
            self.done.append(False)
        self.observation.append(next_state)      
        next_state = tf.convert_to_tensor(next_state,dtype=tf.float32)
        next_state_value_estimate = v_nn(next_state).numpy()[0] 
        self.val.append(next_state_value_estimate[0])
        print(self.rew.get(t))
        print( self.val.get(t + 1))
        print( self.val.get(t))
        if  episode_done:
            self.delta.append(self.rew.get(t)  - self.val.get(t))
        else:
            self.delta.append(self.rew.get(t) + gamma * self.val.get(t + 1)- self.val.get(t))
        self.last_observation=next_state
    
    def calc_advantages(self,ending_time,horizon):
        advantages=[]
        values = []
        cumulative_advantage = 0
        last_value_sample = self.val.get(ending_time)
        for i in range(horizon):
            if self.done.get(ending_time - i - 1):
                cumulative_advantage = 0
                last_value_sample = 0
            cumulative_advantage = self.delta.get(ending_time - i - 1) + (gamma * gae_lambda * cumulative_advantage)
            advantages.append(cumulative_advantage)
            last_value_sample = gamma * last_value_sample + self.rew.get(ending_time - i - 1)
            values.append(last_value_sample)
        advantages.reverse()
        values.reverse()
        for i in range(len(advantages)):
            self.estimate_the_advantages.append(advantages[i])
            self.estimate_the_values.append(values[i])

    def get_data(self,ending_time,horizon):
        return (self.observation.in_range(ending_time - horizon, horizon),
                self.act.in_range(ending_time - horizon, horizon),
                self.policy.in_range(ending_time - horizon, horizon),
                self.estimate_the_advantages.in_range(ending_time - horizon, horizon),
                self.estimate_the_values.in_range(ending_time - horizon, horizon))

    def clear_history(self, ending_time,horizon):
        self.observation.remove_excess(ending_time - horizon - 10)
        self.act.remove_excess(ending_time - horizon - 1)
        self.rew.remove_excess(ending_time - horizon - 1)
        self.val.remove_excess(ending_time - horizon - 1)
        self.policy.remove_excess(ending_time - horizon - 1)
        self.delta.remove_excess(ending_time - horizon - 1)
        self.done.remove_excess(ending_time - horizon - 1)
