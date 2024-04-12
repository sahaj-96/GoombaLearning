from collections import deque
import gym
import cv2
import numpy as np

class EpisodeEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)        
        self.was_real_done = True #check if last episode ended naturally or due to loss of life.

    def step(self, action):
        observ, rwd, done, info = self.env.step(action)        
        if self.env.unwrapped._flag_get: #if set it ensures that agent has reached some specific state and deserves a reward
            rwd += 250
            done = True            
        if self.env.unwrapped._is_dying:# if set indicate agent in a bad state
            rwd -= 80   
            done = True                 
        self.was_real_done = done#check prev episode ended naturally or not
        return observ, rwd, done, info

    def reset(self, **kwargs):#reset if life exauhts keeping all states reachable without the learner needing to know the details.
        if self.was_real_done:
            observ = self.env.reset(**kwargs)
        else:
            observ, _, _, _ = self.env.step(0)    
        return observ

class SclaeRwrd(gym.RewardWrapper):
    def rwrd(self, rwrd):

        return rwrd * 0.1#ensures reward within a spec range for PPO 

class pre_process(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.w = 256
        self.h = 256
        self.obv_space = gym.spaces.Box(low=0, high=255,shape=(self.h, self.w, 1), dtype=np.uint8)

    def obv(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.w,self.h), interpolation=cv2.INTER_AREA)
        frame = frame[:, :, None]

        return frame

