from multiprocessing import Pipe,Process
import tensorflow as tf
import numpy as np
starting_time=0
gamma=.99
gae_lambda=0.99
version=1 # there are for versions u can fill here any of 1,2,3,4
from . import env_maker
version=env_maker.version
env,_,_=env_maker.make_env(env_idx=version-1)
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
    def remove_excess(self,time_interval):
        for_removing=time_interval-self.first_time+1
        if for_removing>0:
            self.list=self.list[for_removing:]
            self.first_time=time_interval+1
    def append(self,to_add):
        self.list.append(to_add)
    def get(self, at_time_t):
        index = at_time_t - self.first_time
        if 0 <= index < len(self.list):
            return self.list[index]
        else:
            return 0
    def future_len(self):
        return len(self.list)
    def in_range(self,from_time,length):
        return self.list[(from_time-self.first_time):(from_time-self.first_time+length)]


class Maincontroller:
    def __init__(self, environment):
        self.env=environment
        self.observation=TimeOrderedList(first_time=starting_time) 
        self.last_observation=tf.convert_to_tensor(np.expand_dims(self.env.reset(),axis=0),dtype=tf.float32)
        self.observation.append(self.last_observation)
        self.act = self.rew = self.val = self.policy = self.delta = self.done = TimeOrderedList(first_time=starting_time)
        self.episode_start_time = 0
        self.rew_ofeach_episode = self.xpos_ofeach_episode = self.rew_ofcurr_episode = self.xpos_ofcurr_episode = []
        self.estimate_the_advantages = self.estimate_the_values = TimeOrderedList(first_time=starting_time)

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
            self.rew_ofcurr_episode.clear()
            self.xpos_ofeach_episode.append(self.xpos_ofcurr_episode)
            self.xpos_ofcurr_episode.clear()
            next_state=self.env.reset()
            next_state=np.expand_dims(next_state,axis=0)
            self.episode_start_time=t+1
        else:
            self.done.append(False)
        self.observation.append(next_state)      
        next_state = tf.convert_to_tensor(next_state,dtype=tf.float32)
        next_state_value_estimate = v_nn(next_state).numpy()[0] 
        self.val.append(next_state_value_estimate[0])  
        self.delta.append(self.rew.get(t) + (1 if episode_done else 0)*gamma*self.val.get(t+1)-self.val.get(t))
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
        self.estimate_the_advantages.extend(reversed(advantages))
        self.estimate_the_values.extend(reversed(values))

    def get_data(self,ending_time,horizon):
        obs_range=[self.observation[ending_time - i] for i in range(horizon,0,-1) if ending_time - i >= 0]
        act_range=[self.act[ending_time - i] for i in range(horizon,0,-1) if ending_time - i >= 0]
        policy_range=[self.policy[ending_time - i] for i in range(horizon,0,-1) if ending_time - i >= 0]
        adv_range=[self.estimate_the_advantages[ending_time - i] for i in range(horizon,0,-1) if ending_time - i >= 0]
        val_range=[self.estimate_the_values[ending_time - i] for i in range(horizon,0,-1) if ending_time - i >= 0]
        return obs_range, act_range, policy_range, adv_range, val_range

    def clear_history(self, ending_time,horizon):
        self.observation.remove_excess(ending_time - horizon - 10)
        self.act.remove_excess(ending_time - horizon - 1)
        self.rew.remove_excess(ending_time - horizon - 1)
        self.val.remove_excess(ending_time - horizon - 1)
        self.policy.remove_excess(ending_time - horizon - 1)
        self.delta.remove_excess(ending_time - horizon - 1)
        self.done.remove_excess(ending_time - horizon - 1)
