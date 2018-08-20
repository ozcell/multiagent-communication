import numpy as np
import scipy as sc
import time
import imageio

import torch as K
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from macomm.algorithms2 import MAMDDPG, MAHCDDPG, MAHDDDPG
from macomm.environments import communication, communication_v2, make_env_cont
from macomm.experience import ReplayMemory, ReplayMemoryComm, Transition, Transition_Comm
from macomm.exploration import OUNoise
from macomm.utils import Saver, Summarizer, get_noise_scale, get_params2, running_mean, intrinsic_reward

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

    GAMMA = config['gamma'] # 0.95
    TAU = config['tau'] # 0.01

    ACTOR_LR = config['plcy_lr'] # 0.01
    CRITIC_LR = config['crtc_lr'] # 0.01

    COMM_ACTOR_LR = config['comm_plcy_lr'] # 0.01
    COMM_CRITIC_LR = config['comm_crtc_lr'] # 0.01

    MEM_SIZE = config['buffer_length'] # 1000000 

    REGULARIZATION = config['regularization'] # True
    NORMALIZED_REWARDS = config['reward_normalization'] # True

    if ENV_NAME == 'waterworld':
        from madrl_environments.pursuit import MAWaterWorld_mod
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
        env = make_env_cont(ENV_NAME, benchmark=BENCHMARK)

    num_agents = env.n
    observation_space = env.observation_space[0].shape[0]

    if ENV_NAME == 'waterworld':
        action_space = env.action_space[0]
        OUT_FUNC = F.tanh
    else:
        action_space = env.action_space[0].n
        OUT_FUNC = F.sigmoid

    #if (config['agent_alg'] == 'MAMDDPG' or config['agent_alg'] == 'MAHCDDPG'):
    medium_space = observation_space      
    #else:
    #    medium_space = observation_space + 1

    env.seed(SEED)
    K.manual_seed(SEED)
    np.random.seed(SEED)
    
    if config['agent_alg'] == 'MAMDDPG':
        MODEL = MAMDDPG
        comm_env = 'autoencoder'
    elif config['agent_alg'] == 'MAHCDDPG':
        MODEL = MAHCDDPG
        comm_env = 'hierarchical'
    elif config['agent_alg'] == 'MAHDDDPG':
        MODEL = MAHDDDPG
        comm_env = 'hierarchical'
    
    discrete_comm = config['discrete_comm'] 
        
    if config['agent_type'] == 'basic':
        from macomm.agents.basic import Actor, Critic
    elif config['agent_type'] == 'deep':
        from macomm.agents.deep import Actor, Critic

    if comm_env == 'autoencoder':
        from macomm.agents.basic import Medium as Comm_Actor
        Comm_Critic = None
    else:
        if config['comm_agent_type'] == 'basic':
            from macomm.agents.basic import Actor as Comm_Actor, Critic as Comm_Critic
        elif config['comm_agent_type']  == 'deep':
            from macomm.agents.deep import Actor as Comm_Actor, Critic as Comm_Critic
    
    if config['verbose'] > 1:
        # utils
        summaries = (Summarizer(config['dir_summary_train'], config['port'], config['resume']),
                    Summarizer(config['dir_summary_test'], config['port'], config['resume']))
        saver = Saver(config)
    else:
        summaries = None
        saver = None


    #exploration initialization
    ounoise = OUNoise(action_space)
    comm_ounoise = OUNoise(1)

    #model initialization
    optimizer = (optim.Adam, (ACTOR_LR, CRITIC_LR, COMM_ACTOR_LR, COMM_CRITIC_LR)) # optimiser func, (actor_lr, critic_lr)
    loss_func = F.mse_loss
    model = MODEL(num_agents, observation_space, action_space, medium_space, optimizer, 
                  Actor, Critic, loss_func, GAMMA, TAU, out_func=OUT_FUNC,
                  discrete=False, regularization=REGULARIZATION, normalized_rewards=NORMALIZED_REWARDS, 
                  communication=comm_env, discrete_comm=discrete_comm, Comm_Actor=Comm_Actor, Comm_Critic=Comm_Critic
                  )
    
    if config['resume'] != '':
        for i, param in enumerate(save_dict['model_params']):
            model.entities[i].load_state_dict(param)
    
    #memory initilization
    if model.communication is None:
        memory = [ReplayMemory(MEM_SIZE)]
    elif model.communication == 'hierarchical':
        memory = [ReplayMemoryComm(MEM_SIZE), ReplayMemoryComm(MEM_SIZE)]
    else:
        memory = [ReplayMemoryComm(MEM_SIZE)]
        
    experiment_args = (env, comm_env, memory, ounoise, comm_ounoise, config, summaries, saver, start_episode)
          
    return model, experiment_args

def to_onehot(actions):
    actions = actions.view(-1)
    onehot = K.zeros_like(actions, dtype=K.uint8)
    onehot[actions.argmax()] = 1
    return onehot

def run(model, experiment_args, train=True):

    total_time_start =  time.time()

    env, comm_env, memory, ounoise, comm_ounoise, config, summaries, saver, start_episode = experiment_args
    
    start_episode = start_episode if train else 0
    NUM_EPISODES = config['n_episodes'] if train else config['n_episodes_test'] 
    EPISODE_LENGTH = config['episode_length'] if train else config['episode_length_test'] 
    
    t = 0
    episode_rewards_all = []
    episode_aux_rewards_all = []

    print('oops changed')
        
    for i_episode in range(start_episode, NUM_EPISODES):
        
        episode_time_start = time.time()
        
        frames = []
        
        # Initialize the environment and state
        observations = np.stack(env.reset())
        observations = K.tensor(observations, dtype=K.float32).unsqueeze(1)

        cumm_rewards = K.zeros((model.num_agents,1,1), dtype=dtype)

        ounoise.scale = get_noise_scale(i_episode, config)
        comm_ounoise.scale = get_noise_scale(i_episode, config)
        
        # monitoring variables
        episode_rewards = np.zeros((model.num_agents,1,1))
        episode_aux_rewards = np.zeros((model.num_agents,1,1))
        
        for i_step in range(EPISODE_LENGTH):

            model.to_cpu()

            if model.communication == 'hierarchical':
                if i_step % config['hierarchical_time_scale']  == 0:
                    observations_init = observations.clone()
                    cumm_rewards = K.zeros((model.num_agents,1,1), dtype=dtype)
                    if config['agent_alg'] == 'MAHCDDPG':
                        if model.discrete_comm:
                            comm_actions = model.select_comm_action(observations_init, True if train else False).unsqueeze(0)
                            #medium = observations_init[K.tensor(comm_actions, dtype=K.uint8)[0,0,]]
                        else:
                            comm_actions = model.select_comm_action(observations_init, comm_ounoise if train else False).unsqueeze(0)
                            #medium = observations_init[(comm_actions > .5)[0,0,:]]
                            #medium = (K.mean(observations_init, dim=0) if medium.shape == K.Size([0]) else K.mean(medium, dim=0)).unsqueeze(0)
                            #medium = (K.mean(observations_init, dim=0) if medium.shape == K.Size([0]) else K.zeros_like(observations_init[0])).unsqueeze(0)
                    elif config['agent_alg'] == 'MAHDDDPG':
                        comm_actions = []
                        for i in range(model.num_agents):
                            comm_action = model.select_comm_action(observations_init[[i], ], i, comm_ounoise if train else False)
                            comm_actions.append(comm_action)
                        comm_actions = K.stack(comm_actions)
                        #medium = observations_init[(comm_actions > .5)[:,0,0]]
                        #medium = (K.mean(observations_init, dim=0) if medium.shape == K.Size([0]) else K.mean(medium, dim=0)).unsqueeze(0)
                    medium = observations_init[to_onehot(comm_actions)]

            else:
                if config['agent_alg'] == 'MAMDDPG':
                    medium = model.select_comm_action(observations).unsqueeze(0)

            actions = []
            for i in range(model.num_agents):
                action = model.select_action(K.cat([observations[[i], ], medium], dim=-1), i, ounoise if train else False)
                actions.append(action)
            actions = K.stack(actions)

            next_observations, rewards, dones, infos = env.step(actions.squeeze(1))
            next_observations = K.tensor(next_observations, dtype=dtype).unsqueeze(1)
            rewards = K.tensor(rewards, dtype=dtype).view(-1,1,1)
            
            # different types od aux reward to train the second policy
            intr_rewards = intrinsic_reward(env, medium.numpy())
            intr_rewards = K.tensor(intr_rewards, dtype=dtype).view(-1,1,1)
            cumm_rewards += rewards

            if config['aux_reward_type'] == 'intrinsic':
                aux_rewards = intr_rewards
            elif config['aux_reward_type'] == 'cummulative':
                aux_rewards = rewards

            # for monitoring
            episode_rewards += rewards
            episode_aux_rewards += aux_rewards

            # if it is the last step we don't need next obs
            if i_step == EPISODE_LENGTH-1:
                next_observations = None

            # Store the transition in memory
            if train:
                memory[0].push(observations, actions, next_observations, aux_rewards, medium, None, None, None, None)
                if model.communication == 'hierarchical' and (i_step+1) % config['hierarchical_time_scale'] == 0:
                    memory[1].push(observations_init, None, next_observations, cumm_rewards, medium, comm_actions, None, None, None)

            # Move to the next state
            observations = next_observations
            t += 1
            
            # Use experience replay and train the model
            critic_losses = None
            actor_losses = None
            medium_loss = None
            if train:
                if (sum([True for i_memory in memory if len(i_memory) > config['batch_size']-1]) == len(memory) and t%config['steps_per_update'] == 0):
                    model.to_cuda()   
                    critic_losses = []
                    actor_losses = []            
                    for i in range(env.n):
                        batch = Transition_Comm(*zip(*memory[0].sample(config['batch_size'])))
                        if model.communication == 'hierarchical':
                            batch2 = Transition_Comm(*zip(*memory[1].sample(config['batch_size'])))
                            critic_loss, actor_loss = model.update_parameters(batch, batch2, i)
                        else:
                            critic_loss, actor_loss, medium_loss = model.update_parameters(batch, i)
                        critic_losses.append(critic_loss)
                        actor_losses.append(actor_loss)
                        
            # Record frames
            if config['render'] > 0 and i_episode % config['render'] == 0:
                if config['env_id'] == 'waterworld':
                    frames.append(sc.misc.imresize(env.render(), (300, 300)))
                else:
                    frames.append(env.render(mode='rgb_array')[0]) 
                    if config['render_color_change']: 
                        for i_geoms, geoms in enumerate(env.render_geoms):
                            if i_geoms == env.world.leader:
                                geoms.set_color(0.85,0.35,0.35,0.55)
                            else:
                                geoms.set_color(0.35,0.35,0.85,0.55)
                            if i_geoms == env.n - 1:
                                break

        # <-- end loop: i_step 
        
        ### MONITORIRNG ###

        episode_rewards_all.append(episode_rewards.sum())
        episode_aux_rewards_all.append(episode_aux_rewards.sum())
        if config['verbose'] > 0:
        # Printing out
            if (i_episode+1)%100 == 0:
                print("==> Episode {} of {}".format(i_episode + 1, NUM_EPISODES))
                print('  | Id exp: {}'.format(config['exp_id']))
                print('  | Exp description: {}'.format(config['exp_descr']))
                print('  | Env: {}'.format(config['env_id']))
                print('  | Process pid: {}'.format(config['process_pid']))
                print('  | Tensorboard port: {}'.format(config['port']))
                print('  | Episode total reward: {}'.format(episode_rewards.sum()))
                print('  | Running mean of total reward: {}'.format(running_mean(episode_rewards_all)[-1]))
                print('  | Running mean of total comm_reward: {}'.format(running_mean(episode_aux_rewards_all)[-1]))
                print('  | Medium loss: {}'.format(medium_loss))
                print('  | Time episode: {}'.format(time.time()-episode_time_start))
                print('  | Time total: {}'.format(time.time()-total_time_start))
                        
        if config['verbose'] > 0:    
            ep_save = i_episode+1 if (i_episode == NUM_EPISODES-1) else None  
            is_best_save = None
            is_best_avg_save = None   
                
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
    
            if (i_episode+1)%100 == 0:
                summary = summaries[0] if train else summaries[1]
                summary.update_log(i_episode, 
                                episode_rewards.sum(), 
                                list(episode_rewards.reshape(-1,)), 
                                critic_loss        = critic_losses, 
                                actor_loss         = actor_losses,
                                to_save            = to_save, 
                                comm_reward_total  = episode_aux_rewards.sum(),
                                comm_reward_agents = list(episode_aux_rewards.reshape(-1,))
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
    
    return (episode_rewards_all, episode_aux_rewards_all)


if __name__ == '__main__':

    monitor_macddpg_p2 = []
    monitor_macddpg_p2_test = []
    for i in range(0,5):
        config = get_params(args=['--exp_id','MACDDPG_P2_120K_'+ str(i+1), 
                                '--random_seed', str(i+1), 
                                '--agent_alg', 'MACDDPG',
                                '--protocol_type', str(2),
                                '--n_episodes', '120000',
                                '--verbose', '2',
                                ]
                        )
        model, experiment_args = init(config)

        env, comm_env, memory, ounoise, comm_ounoise, config, summaries, saver, start_episode = experiment_args

        tic = time.time()
        monitor = run(model, experiment_args, train=True)
        monitor_test = run(model, experiment_args, train=False)

        toc = time.time()

        env.close()
        for summary in summaries:
            summary.close()
            
        monitor_macddpg_p2.append(monitor)
        monitor_macddpg_p2_test.append(monitor_test)
        
        np.save('./monitor_macddpg_p2.npy', monitor_macddpg_p2)
        np.save('./monitor_macddpg_p2_test.npy', monitor_macddpg_p2_test)
        
        print(toc-tic)