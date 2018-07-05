import numpy as np
import scipy as sc
import time
import imageio

import torch as K
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from macomm.algorithms import MADDPG, MACDDPG, DDPG
from macomm.environments import communication, make_env_cont
from macomm.experience import ReplayMemory, ReplayMemoryComm, Transition, Transition_Comm
from macomm.exploration import OUNoise
from macomm.utils import Saver, Summarizer, get_noise_scale, get_params, running_mean

from madrl_environments.pursuit import MAWaterWorld_mod

import pdb


device = K.device("cuda" if K.cuda.is_available() else "cpu")
dtype = K.float32


def init(config):
    
    if config['resume'] != '':
        resume_path = config['resume']
        saver = Saver(config)
        config, start_episode, save_dict = saver.resume_ckpt()
        config['resume'] = resume_path
    else:
        start_episode = 0
        
    #hyperparameters
    ENV_NAME = config['env_id'] #'simple_spread'
    BENCHMARK = config['benchmark'] 
    SEED = config['random_seed'] # 1

    BATCH_SIZE = config['batch_size'] # 1024
    GAMMA = config['gamma'] # 0.95
    TAU = config['tau'] # 0.01

    ACTOR_LR = config['plcy_lr'] # 0.01
    CRITIC_LR = config['crtc_lr'] # 0.01

    MAX_EPISODE_LENGTH = config['episode_length'] # 25
    STEPS_PER_UPDATE = config['steps_per_update'] # 100
    NUM_EPISODES = config['n_episodes'] # 60000

    MEM_SIZE = config['buffer_length'] # 1000000 

    INIT_NOISE = config['init_noise_scale'] # 0.3
    FINAL_NOISE = config['final_noise_scale'] # 0.0
    EXP_END = config['n_exploration_eps'] # NUM_EPISODES #60000

    DISCRETE = config['discrete_action'] # False
    REGULARIZATION = config['regularization'] # True
    NORMALIZED_REWARDS = config['reward_normalization'] # True

    PROTOCOL = config['protocol_type'] # 2
    CONS_LIM = config['consecuitive_limit'] # 5

    if ENV_NAME == 'waterworld':
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
        if DISCRETE:
            env = make_env_disc(ENV_NAME)
        else:
            env = make_env_cont(ENV_NAME, benchmark=BENCHMARK)

    num_agents = env.n
    observation_space = env.observation_space[0].shape[0]
    medium_space = observation_space + 1
    if ENV_NAME == 'waterworld':
        action_space = env.action_space[0]
    else:
        action_space = env.action_space[0].n

    env.seed(SEED)
    K.manual_seed(SEED)
    np.random.seed(SEED)
    
    if config['agent_alg'] == 'MACDDPG':
        MODEL = MACDDPG
        comm_env = communication(protocol_type=PROTOCOL, 
                                 consecuitive_limit=CONS_LIM, 
                                 num_agents=num_agents)
    elif config['agent_alg'] == 'MADDPG':
        MODEL = MADDPG
        comm_env = None
    elif config['agent_alg'] == 'DDPG':
        MODEL = DDPG
        comm_env = None

    if config['agent_type'] == 'basic':
        from macomm.agents.basic import Actor, Critic
    elif config['agent_type'] == 'deep':
        from macomm.agents.deep import Actor, Critic
    
    # utils
    summaries = (Summarizer(config['dir_summary_train'], config['port'], config['resume']),
                 Summarizer(config['dir_summary_test'], config['port'], config['resume']))
                 
    
    saver = Saver(config)

    #exploration initialization
    ounoise = OUNoise(action_space)
    comm_ounoise = OUNoise(1)

    #model initialization
    optimizer = (optim.Adam, (ACTOR_LR, CRITIC_LR)) # optimiser func, (actor_lr, critic_lr)
    loss_func = F.mse_loss
    model = MODEL(num_agents, observation_space, action_space, medium_space, optimizer, 
                  Actor, Critic, loss_func, GAMMA, TAU, 
                  discrete=DISCRETE, regularization=REGULARIZATION, normalized_rewards=NORMALIZED_REWARDS, 
                  communication=comm_env
                  )
    
    if config['resume'] != '':
        for i, param in enumerate(save_dict['model_params']):
            model.entities[i].load_state_dict(param)
    
    #memory initilization
    if model.communication is not None:
        memory = ReplayMemoryComm(MEM_SIZE)
    else:
        memory = ReplayMemory(MEM_SIZE)
        
    experiment_args = (env, comm_env, memory, ounoise, comm_ounoise, config, summaries, saver, start_episode)
          
    return model, experiment_args

def run(model, experiment_args, train=True):

    total_time_start =  time.time()

    env, comm_env, memory, ounoise, comm_ounoise, config, summaries, saver, start_episode = experiment_args
    
    start_episode = start_episode if train else 0
    NUM_EPISODES = config['n_episodes'] if train else config['n_episodes_test'] 
    
    reward_best = float("-inf")
    avg_reward_best = float("-inf")
    
    t = 0
    episode_rewards_all = []
    episode_comm_rewards_all = []
        
    for i_episode in range(start_episode, NUM_EPISODES):
        
        episode_time_start = time.time()
        
        frames = []
        
        # Initialize the environment and state
        observations = np.stack(env.reset())
        observations = K.tensor(observations, dtype=K.float32).unsqueeze(1)
        #comm_env.reset()

        episode_rewards = np.zeros((model.num_agents,1,1))
        episode_comm_rewards = np.zeros((model.num_agents,1,1))
        
        ounoise.scale = get_noise_scale(i_episode, config)
        comm_ounoise.scale = get_noise_scale(i_episode, config)

        for i_step in range(config['episode_length']):

            model.to_cpu()
            
            if model.communication is not None:
                comm_actions = []
                for i in range(model.num_agents): 
                    comm_action = model.select_comm_action(observations[[i], ], i, comm_ounoise if train else False)
                    comm_actions.append(comm_action)
                comm_actions = K.stack(comm_actions).cpu()

                medium, comm_rewards = comm_env.step(observations, comm_actions)
                medium = K.tensor(medium, dtype=K.float32)
                comm_rewards = K.tensor(comm_rewards, dtype=dtype).view(-1,1,1)
                episode_comm_rewards += comm_rewards 

            actions = []
            for i in range(model.num_agents):
                if model.discrete:
                    if model.communication is not None:
                        action = model.select_action(K.cat([observations[[i], ], medium], dim=-1), i, train)
                    else:
                        action = model.select_action(observations[[i], ], i, train)
                else:
                    if model.communication is not None:
                        action = model.select_action(K.cat([observations[[i], ], medium], dim=-1), i, ounoise if train else False)
                    else:
                        action = model.select_action(observations[[i], ], i, ounoise if train else False)
                actions.append(action)
            actions = K.stack(actions).cpu()

            next_observations, rewards, dones, infos = env.step(actions.squeeze(1))
            next_observations = K.tensor(next_observations, dtype=K.float32).unsqueeze(1)
            rewards = K.tensor(rewards, dtype=dtype).view(-1,1,1)
            episode_rewards += rewards      

            # if it is the last step we don't need next obs
            if i_step == config['episode_length']-1:
                next_observations = None

            # Store the transition in memory
            if train:
                if model.communication is not None:
                    memory.push(observations, actions, next_observations, rewards, medium, comm_actions, comm_rewards)
                else:
                    memory.push(observations, actions, next_observations, rewards)

            # Move to the next state
            observations = next_observations
            t += 1
            
            # Use experience replay and train the model
            critic_losses = None
            actor_losses = None
            if train:
                if len(memory) > config['batch_size']-1 and t%config['steps_per_update'] == 0:
                    critic_losses = []
                    actor_losses = []                  
                    for i in range(env.n):
                        if model.communication is not None:
                            batch = Transition_Comm(*zip(*memory.sample(config['batch_size'])))
                        else:
                            batch = Transition(*zip(*memory.sample(config['batch_size'])))
                        model.to_cuda()
                        critic_loss, actor_loss = model.update_parameters(batch, i)
                        critic_losses.append(critic_loss)
                        actor_losses.append(actor_loss)
                        
            # Record frames
            if config['render'] > 0 and i_episode % config['render'] == 0:
                if config['env_id'] == 'waterworld':
                    frames.append(sc.misc.imresize(env.render(), (300, 300)))
                else:
                    frames.append(env.render(mode='rgb_array'))  

        # <-- end loop: i_step 
        
        ### MONITORIRNG ###
        
        # Printing out
        episode_rewards_all.append(episode_rewards.sum())
        episode_comm_rewards_all.append(episode_comm_rewards.sum())
        if (i_episode+1)%100 == 0:
            print("==> Episode {} of {}".format(i_episode + 1, NUM_EPISODES))
            print('  | Id exp: {}'.format(config['exp_id']))
            print('  | Exp description: {}'.format(config['exp_descr']))
            print('  | Env: {}'.format(config['env_id']))
            print('  | Process pid: {}'.format(config['process_pid']))
            print('  | Tensorboard port: {}'.format(config['port']))
            print('  | Episode total reward: {}'.format(episode_rewards.sum()))
            print('  | Running mean of total reward: {}'.format(running_mean(episode_rewards_all)[-1]))
            print('  | Running mean of total comm_reward: {}'.format(running_mean(episode_comm_rewards_all)[-1]))
            print('  | Time episode: {}'.format(time.time()-episode_time_start))
            print('  | Time total: {}'.format(time.time()-total_time_start))
            
        # Best reward so far?
        if episode_rewards.sum() >= reward_best:
            reward_best = episode_rewards.sum()
            is_best = True
        else:
            is_best = False
        # Best average reward so far?
        if (i_episode > 100) and running_mean(episode_rewards_all)[-1] > avg_reward_best:
            avg_reward_best = running_mean(episode_rewards_all)[-1]
            is_best_avg = True 
        else:
            is_best_avg = False
                    
        # Update logs and save model                     
        ep_save = i_episode+1 if (i_episode % config['save_epochs'] == 0 or i_episode == config['n_episodes']-1) else None
        is_best_save = reward_best if is_best else None
        is_best_avg_save = avg_reward_best if is_best_avg else None   
            
        if (not train) or ((np.asarray([ep_save, is_best_save, is_best_avg_save]) == None).sum() == 3):
            to_save = False
        else:
            model.to_cpu()
            saver.save_checkpoint(save_dict   = {'model_params': [entity.state_dict() for entity in model.entities]},
                                  episode     = ep_save,
                                  is_best     = is_best_save,
                                  is_best_avg = is_best_avg_save
                                  )
            to_save = True
        
        summary = summaries[0] if train else summaries[1]
        summary.update_log(i_episode, 
                           episode_rewards.sum(), 
                           list(episode_rewards.reshape(-1,)), 
                           critic_loss = critic_losses, 
                           actor_loss = actor_losses,
                           to_save = to_save, 
                           comm_reward_total = episode_comm_rewards.sum(),
                           comm_reward_agents = list(episode_comm_rewards.reshape(-1,))
                          )
        

        # Save gif
        dir_monitor = config['dir_monitor_train'] if train else config['dir_monitor_test']
        if config['render'] > 0 and i_episode % config['render'] == 0:
            if config['env_id'] == 'waterworld':
                imageio.mimsave('{}/{}.gif'.format(dir_monitor, i_episode), frames[0::3])
            else:
                imageio.mimsave('{}/{}.gif'.format(dir_monitor, i_episode), frames)
            
    # <-- end loop: i_episode
    if train:
        print('Training completed')
    else:
        print('Test completed')
    
    return (episode_rewards_all, episode_comm_rewards_all)

if __name__ == '__main__':

    config = get_params(args=['--exp_id','wtf', '--agent_alg', 'MACDDPG'])
    model, experiment_args = init(config)
    
    env, comm_env, memory, ounoise, comm_ounoise, config, summaries, saver, start_episode = experiment_args
    
    tic = time.time()
    monitor = run(model, args, train=True)
    monitor_test = run(model, args, train=False)
    
    toc = time.time()
    
    env.close()
    for summary in summaries:
        summary.close()
    
    print(toc-tic)