import gym
from collections import deque
import numpy as np
import cv2

class EpisodeEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)        
        self.was_real_done = True #check if last episode ended naturally or due to loss of life.

    def step(self, action):
        observ, rwd, done, info = self.env.step(action)        
        if self.env.unwrapped._flag_get: #if set it ensures that agent has reached some specific state and deserves a reward
            rwd += 150
            done = True            
        if self.env.unwrapped._is_dying:# if set indicate agent in a bad state
            rwd -= 50   
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
    def reward(self, rwrd):

        return rwrd * 0.05#ensures reward within a spec range for PPO 

class pre_process(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.w = 96
        self.h = 96
        self.obv_space = gym.spaces.Box(low=0, high=255,shape=(self.h, self.w, 1), dtype=np.uint8)

    def observation(self, frame):
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

class scaleobsframe(gym.ObservationWrapper): # ensure observations are within a reasonable range for NN.
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space=gym.spaces.Box(low=0, high=1, shape=env.observation_space.shape, dtype=np.float32)

    def observation(self, observ):
        return np.array(observ).astype(np.float32) / 255.0
    

class Stackframe(gym.Wrapper):
    def __init__(self, env, n):
        gym.Wrapper.__init__(self, env)
        self.n=n #n number of frames to stack
        self.frames=deque([], maxlen=n)
        shapes=env.observation_space.shape
        self.observation_space=gym.spaces.Box(low=0, high=255, shape=(shapes[:-1] + (shapes[-1] * n,)), dtype=env.observation_space.dtype)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.n):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.n
        return LazyFrames(list(self.frames))
    

class LazyFrames():# ensures common frames b/w observations are only stored once.
    def __init__(self, nframe):
        self.frames=nframe
        self.cache=None#for caching

    def concframes(self):
        if self.cache is None:
            self.cache=np.concatenate(self.frames, axis=-1)
            self.frames=None
        return self.cache

    def __array__(self, dtype=None):
        z=self.concframes()
        if dtype is not None:
            z=z.astype(dtype)
        return z

    def __len__(self):
        return len(self.concframes())

    def __getitem__(self, i):
        return self.concframes()[i]

    def count(self):
        f=self.concframes()
        return f.shape[f.ndim - 1]

    def frame(self, i):
        return self.concframes()[..., i]