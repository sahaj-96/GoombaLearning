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

class ScaleRwrd(gym.RewardWrapper):
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

class frameskipper(gym.Wrapper):
    def __init__(self, env, nofframes, stickprob):
        gym.Wrapper.__init__(self, env)
        self.nofframes = nofframes
        self.stickprob = stickprob
        self.curr_action = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")#flag i dicates whether env should render itself or not

    def reset(self, **kwargs):
        self.curr_action = None
        return self.env.reset(**kwargs)# reset wrapped env to its initial state and return initial obv

    def step(self, action):
        done = False
        total_reward = 0
        for i in range(self.nofframes):
            if self.curr_action is None:#if first step
                self.curr_action = action
            elif i==0:# first substep of a frame
                if self.rng.rand() > self.stickprob:#to add stochasticity by occasionally sticking with same action.
                    self.curr_action = action
            elif i==1:
                self.curr_action = action
            if self.supports_want_render and i<self.nofframes-1:
                observ, rew, done, info = self.env.step(self.curr_action, want_render=False)
            else:
                observ, rew, done, info = self.env.step(self.curr_action)
            total_reward += rew
            if done: break
        return observ, total_reward, done, info

    def seed(self, s):
        self.rng.seed(s) 