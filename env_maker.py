from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from . import preprocess
def make_env(env_idx):
    dict=[
        {'state':'SuperMarioBros-1-1-v0'},
        {'state':'SuperMarioBros-1-2-v0'},
        {'state':'SuperMarioBros-1-3-v0'},
        {'state':'SuperMarioBros-2-2-v0'},      
    ]
    env=gym_super_mario_bros.make(id=dict[env_idx]['state'])
    env=JoypadSpace(env,COMPLEX_MOVEMENT)
    env = preprocess.EpisodeEnv(env)
    env = preprocess.ScaleRwrd(env)
    env = preprocess.pre_process(env)
    env = preprocess.frameskipper(env,nofframes=4,stickprob=0.5)
    env = preprocess.scaleobsframe(env)
    env = preprocess.Stackframe(env, 4)
    return env
