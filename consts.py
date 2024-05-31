from . import env_maker
version=1
env = env_maker.make_env(version-1)
observation_shape=env.observation_space.shape
num_actions=env.action_space.n
num_actors=4
gae_lambda = 0.95
gamma = 0.99 
base_clip_epsilon = 0.2
max_steps = 1000000
base_learning_rate = 2.5e-5
horizon = 10240
batch_size = 1024
value_loss_coefficient = .01
entropy_loss_coefficient = .01
gradient_max = 10.0
start_t = 0
