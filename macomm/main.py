import numpy as np
import skimage.io as io
import imageio
import matplotlib.pylab as plt
from collections import namedtuple

import torch as K
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as T

from macomm.agents import Actor, Critic
from macomm.exploration import gumbel_softmax, OUNoise
from macomm.experience import ReplayMemory, ReplayMemoryComm, Transition, Transition_Comm
from macomm.algorithms import MADDPG, MACDDPG
from macomm.environments import communication, make_env_cont

from torchviz import make_dot

device = K.device("cuda" if K.cuda.is_available() else "cpu")
dtype = K.float32

#hyperparameters
env_name = 'simple_spread'
action_space = 2
discrete = False
regularization = True
normalized_rewards = True

if env_name == 'waterworld':
    env = MAWaterWorld_mod(n_pursuers=2, 
                       n_evaders=50,
                       n_poison=50, 
                       obstacle_radius=0.04,
                       food_reward=10.,
                       poison_reward=-1.,
                       ncounter_reward=0.01,
                       n_coop=2,
                       sensor_range=0.2, 
                       obstacle_loc=None, )
    # just for compatibility
    env.n = env.n_pursuers
    env.observation_space = []
    env.action_space = []
    for i in range(env.n):
        env.observation_space.append(np.zeros(213))
        env.action_space.append(2)
else:
    if discrete:
        env = make_env_disc(env_name)
    else:
        env = make_env_cont(env_name, benchmark=True)
        
num_agents = env.n
observation_space = env.observation_space[0].shape[0]
medium_space = observation_space + 1
if env_name == 'waterworld':
    action_space = env.action_space[0]
else:
    action_space = env.action_space[0].n
    
comm_env = communication(5)
        
env.seed(1)
K.manual_seed(1)
np.random.seed(1)

optimizer = (optim.Adam, (0.01, 0.01)) # optimiser func, (actor_lr, critic_lr)
loss_func = F.mse_loss
GAMMA = 0.95
TAU = 0.01
MAX_EPISODE_LENGTH = 25
BATCH_SIZE = 1024
STEPS_PER_UPDATE = 100
NUM_EPISODES = 60000
MEM_SIZE = 1000000

noise_scale = 0.3
final_noise_scale = 0.0
exploration_end = NUM_EPISODES #60000
ounoise = OUNoise(action_space)
comm_ounoise = OUNoise(1)



#memory initilization
memory = ReplayMemoryComm(MEM_SIZE)

#model initialization
model = MACDDPG(num_agents, observation_space, action_space, medium_space, optimizer, 
               loss_func, GAMMA, TAU, 
               discrete=discrete, regularization=regularization, normalized_rewards=normalized_rewards)

closs_all = []
aloss_all = []
t = 0
episode_rewards = []
communication_rewards = []
communication_rewards_all = []

def plot_durations(durations):
    plt.figure(2)
    plt.clf()
    durations_t = K.tensor(durations, dtype=K.float32)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = K.cat((K.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated

for i_episode in range(NUM_EPISODES):
    # Initialize the environment and state
    observations = np.stack(env.reset())
    observations = K.tensor(observations, dtype=K.float32).unsqueeze(1)
    
    episode_reward = np.zeros((num_agents,1,1)) 
    communication_reward = np.zeros((num_agents,1,1))
    
    ounoise.scale = (0.3 - 0.0) * (max(0, exploration_end - i_episode) / exploration_end) + 0.0
    ounoise.reset()
    
    comm_ounoise.scale = (0.3 - 0.0) * (max(0, exploration_end - i_episode) / exploration_end) + 0.0
    comm_ounoise.reset()
    
    for i_step in range(MAX_EPISODE_LENGTH):
        
        comm_actions = []
        for i in range(num_agents): 
            comm_action = model.select_comm_action(observations[[i], ], i, comm_ounoise)
            comm_actions.append(comm_action)
        comm_actions = K.stack(comm_actions).cpu()
        medium, comm_rewards = comm_env.step(observations, comm_actions)
        medium = K.tensor(medium, dtype=K.float32)
        comm_rewards = K.tensor(comm_rewards, dtype=dtype).view(-1,1,1)
        communication_reward += comm_rewards 
        
        actions = []
        for i in range(num_agents): 
            if discrete:
                action = model.select_action(K.cat([observations[[i], ], medium], dim=-1), i, True)
            else:
                action = model.select_action(K.cat([observations[[i], ], medium], dim=-1), i, ounoise)
            actions.append(action)
        actions = K.stack(actions).cpu()
                
        next_observations, rewards, dones, _ = env.step(actions.squeeze(1))
        next_observations = K.tensor(next_observations, dtype=K.float32).unsqueeze(1)
        rewards = K.tensor(rewards, dtype=dtype).view(-1,1,1)
        episode_reward += rewards      
        
        # if it is the last step we don't need next obs
        # if i_step == MAX_EPISODE_LENGTH or dones:
        if i_step == MAX_EPISODE_LENGTH:
            next_observations = None
            
        # Store the transition in memory
        memory.push(observations, actions, next_observations, rewards, medium, comm_actions, comm_rewards)

        # Move to the next state
        observations = next_observations
        t += 1
        #use experience replay
        if len(memory) > BATCH_SIZE-1 and t%STEPS_PER_UPDATE == 0:
            for i in range(env.n):
                batch = Transition_Comm(*zip(*memory.sample(BATCH_SIZE)))
                closs, aloss = model.update_parameters(batch, i)
                if i == 0:
                    closs_all.append(closs)
                    aloss_all.append(aloss)

        if i_step == MAX_EPISODE_LENGTH-1:
            episode_rewards.append(episode_reward[0,0,0])
            communication_rewards.append(communication_reward[0,0,0])
            communication_rewards_all.append(communication_reward)
            if i_episode%100 == 0:
                plot_durations(episode_rewards)           
                plot_durations(communication_rewards)
            break
                
        #if i_episode%100 == 0:
        #    env.render()
        
print('complete')
#env.close()