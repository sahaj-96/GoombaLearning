from . import env_maker
version=1
env,observation_shape,num_actions = env_maker.make_env(version-1)
num_actors = 8
gae_lambda = 0.95
gamma = 0.99 
base_clip_epsilon = 0.2
max_steps = 1e7
base_learning_rate = 2.5e-5
horizon = 128
batch_size = 32
optim_epochs = 5
value_loss_coefficient = .01
entropy_loss_coefficient = .01
gradient_max = 10.0
start_t = 0
