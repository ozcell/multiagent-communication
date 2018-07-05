import matplotlib
matplotlib.use('agg')
import numpy as np
import os, sys, threading, shutil, argparse, json, time, datetime, pickle
import pdb
import matplotlib.pylab as plt

from pathlib import Path
from tensorboardX import SummaryWriter

import torch as K


def running_mean(durations, threshold=100):
    durations_t = K.tensor(durations, dtype=K.float32)
    # take 100 episode averages and plot them too
    if len(durations_t) >= threshold:
        means = durations_t.unfold(0, threshold, 1).mean(1).view(-1)
        means = K.cat((K.zeros(threshold-1), means)).numpy()
    return means

def get_usage_freqs(model, env, comm_env, nb_episodes=400, nb_steps=25):

    usage_freqs = []

    for i_episode in range(nb_episodes):
        observations = np.stack(env.reset())
        observations = K.tensor(observations, dtype=model.dtype).unsqueeze(1)
        for i_step in range(nb_steps):
            comm_actions = []
            for i in range(model.num_agents): 
                comm_action = model.select_comm_action(observations[[i], ], i, False)
                comm_actions.append(comm_action)
            comm_actions = K.stack(comm_actions).cpu()

            medium, _ = comm_env.step(observations, comm_actions)
            medium = K.tensor(medium, dtype=K.float32)
            usage_freqs.append(medium[0,0,-1].item())

            actions = []
            for i in range(model.num_agents): 
                action = model.select_action(K.cat([observations[[i], ], medium], dim=-1), i, False)
                actions.append(action)
            actions = K.stack(actions).cpu()

            next_observations, _, _, _ = env.step(actions.squeeze(1))
            next_observations = K.tensor(next_observations, dtype=K.float32).unsqueeze(1)     
            observations = next_observations
            
    return usage_freqs

def get_noise_scale(i_episode, config):
    
    coef = (config['init_noise_scale'] - config['final_noise_scale'])
    decay = max(0, config['n_exploration_eps'] - i_episode) / config['n_exploration_eps']
    offset = config['final_noise_scale']
    
    scale = coef * decay + offset
    
    return scale

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


def get_params(args=[], verbose=False):

    print("\n\n\n\n")
    print("==============================")
    print("Acquiring Arguments")
    print("==============================")


    parser = argparse.ArgumentParser(description='Arguments')

    # positional
    parser.add_argument("--env_id", default='simple_spread', help="Name of environment")

    # general settings
    parser.add_argument('--random_seed', default=1,type=int,
                        help='random seed for repeatability')
    parser.add_argument("--buffer_length", default=int(1e6), type=int,
                        help="replay memory buffer capacity")
    parser.add_argument("--n_episodes", default=60000, type=int,
                        help="number of episodes")
    parser.add_argument("--n_episodes_test", default=1000, type=int,
                        help="number of episodes")
    parser.add_argument("--episode_length", default=25, type=int,
                        help="number of steps for episode")
    parser.add_argument("--steps_per_update", default=100, type=int,
                        help="target networks update frequency")
    parser.add_argument("--batch_size", default=1024, type=int,
                        help="batch size for model training")
    parser.add_argument("--n_exploration_eps", default=-1, type=int,
                        help="exploration epsilon, -1: n_episodes")
    parser.add_argument("--init_noise_scale", default=0.3, type=float,
                        help="noise initialization")
    parser.add_argument("--final_noise_scale", default=0.0, type=float,
                        help="noise stop updates value")
    parser.add_argument("--save_epochs", default=5000, type=int,
                        help="save model interval")
    parser.add_argument("--plcy_lr", default=0.01, type=float,
                        help="learning rate")
    parser.add_argument("--crtc_lr", default=0.01, type=float,
                        help="learning rate")
    parser.add_argument("--tau", default=0.01, type=float,
                        help="soft update parameter")
    parser.add_argument("--gamma", default=0.95, type=float,
                        help="discount factor")
    parser.add_argument("--agent_alg",
                        default="MACDDPG", type=str,
                        choices=['MACDDPG', 'MADDPG', 'DDPG'])
    parser.add_argument("--device", default='cuda',
                        choices=['cpu','cuda'], 
                        help="device type")
    parser.add_argument("--plcy_hidden_dim", default=128, type=int, 
                        help="actor hidden state dimension")
    parser.add_argument("--crtc_hidden_dim", default=64, type=int, 
                        help="critic hidden state dimension")      
    parser.add_argument("--protocol_type", default=2, type=int, 
                        help="communication protocol type")
    parser.add_argument("--consecuitive_limit", default=5, type=int, 
                        help="number of allowed consecuitive writings")
    #parser.add_argument("--mem_dim", default=4, type=int, 
    #                    help="shared memory dimension")
    #parser.add_argument("--agent_type", default='memory_water',
    #                    choices=['all_1mlp', 'all_2mlp', 
    #                             'a_2mlp_c_1mlp', 'a_water_c_water',
    #                             'memory_water'], help="agent type")

    parser.add_argument("--agent_type", default='simple',
                        choices=['complex'], help="agent type")

    # path arguments
    parser.add_argument('--exp_id', default='no_id',
                        help='experiment ID')
    parser.add_argument('--dir_base', default='./experiments',
                        help='path of the experiment directory')
    parser.add_argument('--port', default=0, type=int,\
                         help='tensorboardX port')
    parser.add_argument('--exp_descr', default='',
                         help='short experiment description')

    # experiment modality
    parser.add_argument('--resume', default='',
                        help='path in case resume is needed')
    parser.add_argument('--expmode', default='normal',
                        help='fast exp mode is usually used to try is code run')
    parser.add_argument("--render", default=0, type=int,
                        help="epochs interval for rendering, 0: no rendering")
    parser.add_argument("--benchmark", action="store_true",
                        help="benchmark mode")
    parser.add_argument("--discrete_action", default="False",
                        choices=['True', 'False'],
                        help="discrete actions")
    parser.add_argument("--regularization", default="True",
                        choices=['True', 'False'],
                        help="Applying regulation to action preactivations")
    parser.add_argument("--reward_normalization", default="True",
                        choices=['True', 'False'],
                        help="Normalizing the rewards")


    # acquire in a dict
    config = parser.parse_args(args)
    args   = vars(config)

    # set arguments which need dependencies
    dir_exp_name = '{}_{}_{}_{}'.format(str([datetime.date.today()][0]),
                                  args['env_id'],
                                  args['agent_type'],
                                  args['exp_id'])

    args['dir_exp'] = '{}/{}'.format(args['dir_base'],dir_exp_name)
    args['dir_saved_models'] = '{}/saved_models'.format(args['dir_exp'])
    args['dir_summary_train'] = '{}/summary/train'.format(args['dir_exp'])
    args['dir_summary_test'] = '{}/summary/test'.format(args['dir_exp'])
    args['dir_monitor_train'] = '{}/monitor/train'.format(args['dir_exp'])
    args['dir_monitor_test'] = '{}/monitor/test'.format(args['dir_exp'])
    # get current process pid
    args['process_pid'] = os.getpid()

    # creating folders:
    directory = args['dir_exp']
    if os.path.exists(directory) and args['resume'] == '':
        toDelete= input("{} already exists, delete it if do you want to continue. Delete it? (yes/no) ".\
            format(directory))

        if toDelete.lower() == 'yes':
            shutil.rmtree(directory)
            print("Directory removed")
        if toDelete == 'No':
            print("It was not possible to continue, an experiment \
                   folder is required.Terminiting here.")
            sys.exit()
    if os.path.exists(directory) == False and args['resume'] == '':
        os.makedirs(directory)
        os.makedirs(args['dir_saved_models'])
        os.makedirs(args['dir_summary_train'])
        os.makedirs(args['dir_summary_test'])
        os.makedirs(args['dir_monitor_train'])
        os.makedirs(args['dir_monitor_test'])

    time.sleep(1)
    with open(os.path.expanduser('{}/arguments.txt'.format(args['dir_exp'])), 'w+') as file:
        file.write(json.dumps(args, indent=4, sort_keys=True))

    if args['expmode'] == 'fast':
        args['batch_size'] = 8
        args['max_episode_len'] = 50

    # noise
    if args['n_exploration_eps'] < 0:
        args['n_exploration_eps'] = args['n_episodes']

    # discrete actions
    if args['discrete_action'] == 'True':
        args['discrete_action'] = True
    else:
        args['discrete_action'] = False

    # discrete actions
    if args['regularization'] == 'True':
        args['regularization'] = True
    else:
        args['regularization'] = False

    # discrete actions
    if args['reward_normalization'] == 'True':
        args['reward_normalization'] = True
    else:
        args['reward_normalization'] = False

    if verbose:
        print("\n==> Arguments:")
        for k,v in sorted(args.items()):
            print('{}: {}'.format(k,v))
        print('\n')


    return args


class Summarizer:
    """
        Class for saving the experiment log files
    """ 
    def __init__(self, path_summary, port, resume=''):

        if resume == '':
            self.__init__from_config(path_summary,port)
        else:
            self.__init__from_file(path_summary,port,resume)


    def __init__from_config(self, path_summary, port):
        self.path_summary = path_summary
        self.writer = SummaryWriter(self.path_summary)
        self.port = port
        self.list_rwd = []
        self.list_comm_rwd = []
        self.list_pkl = []

        if self.port != 0:
            t = threading.Thread(target=self.launchTensorBoard, args=([]))
            t.start()
 

    def __init__from_file(self, path_summary, port, resume):
        
        p = Path(resume).parents[1]
        #print('./{}/summary/log_record.pickle'.format(Path(resume).parents[1]))
        self.path_summary = '{}/summary/'.format(p)
        self.writer = SummaryWriter(self.path_summary)
        self.port = port

        # pdb.set_trace()
        #print(path_summary)
        with open('{}/summary/log_record.pickle'.format(p),'rb') as f:
            pckl = pickle.load(f)
            self.list_rwd = [x['reward_total'] for x in pckl]
            self.list_comm_rwd = [x['comm_reward_total'] for x in pckl]
            self.list_pkl = [x for x in pckl]

        if self.port != 0:
            t = threading.Thread(target=self.launchTensorBoard, args=([]))
            t.start()

    def update_log(self,
        idx_episode, 
        reward_total,
        reward_agents,
        critic_loss = None,
        actor_loss = None,
        to_save=True,
        to_save_plot = 10,
        comm_reward_total = None,
        comm_reward_agents = None,
        ):
            
        self.writer.add_scalar('reward_total',reward_total,idx_episode)

        for a in range(len(reward_agents)):
            self.writer.add_scalar('reward_agent_{}'.format(a), reward_agents[a],idx_episode)
            if critic_loss != None:
                self.writer.add_scalar('critic_loss_{}'.format(a), critic_loss[a],idx_episode)
            if actor_loss != None:
                self.writer.add_scalar('actor_loss_{}'.format(a), actor_loss[a],idx_episode)
            if comm_reward_agents != None:
                self.writer.add_scalar('comm_reward_agents_{}'.format(a), comm_reward_agents[a],idx_episode)

        # save raw values on file
        self.list_rwd.append(reward_total)
        with open('{}/reward_total.txt'.format(self.path_summary), 'w') as fp:
            for el in self.list_rwd:
                fp.write("{}\n".format(round(el, 2)))

        self.list_comm_rwd.append(comm_reward_total)
        with open('{}/comm_reward_total.txt'.format(self.path_summary), 'w') as fp:
            for el in self.list_comm_rwd:
                fp.write("{}\n".format(round(el, 2)))

        # save in pickle format
        dct = {
            'idx_episode'  : idx_episode,
            'reward_total' : reward_total,
            'reward_agents': reward_agents,
            'critic_loss'  : critic_loss,
            'actor_loss'   : actor_loss,
            'comm_reward_total' : comm_reward_total,
            'comm_reward_agents' : comm_reward_agents
        }
        self.list_pkl.append(dct)


        # save things on disk
        if to_save:
            self.writer.export_scalars_to_json(
                '{}/summary.json'.format(self.path_summary))
            with open('{}/log_record.pickle'.format(self.path_summary), 'wb') as fp:
                pickle.dump(self.list_pkl, fp)

        if idx_episode % to_save_plot==0:
            self.plot_fig(self.list_rwd, 'reward_total')
            self.plot_fig(self.list_comm_rwd, 'comm_reward_total')


    def plot_fig(self, record, name):
        durations_t = K.FloatTensor(np.asarray(record))

        fig, ax = plt.subplots( nrows=1, ncols=1, figsize=(20,15))
        ax.grid(True)
        ax.set_ylabel('Duration')
        ax.set_xlabel('Episode')
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
        # plt.yticks(np.arange(-200, 10, 10.0))

        ax.plot(durations_t.numpy())
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = K.cat((K.zeros(99), means))
            ax.plot(means.numpy())

        plt.draw()
        # plt.ylim([-200,10])
        
        fig.savefig('{}/{}.png'.format(self.path_summary,name))
        plt.close(fig)


    def save_fig(self, idx_episode, list_rwd):
        fig, ax = plt.subplots( nrows=1, ncols=1, figsize=(20,15))
        ax.plot(range(idx_episode+1), list_rwd)
        ax.grid(True)
        ax.set_ylabel('Total reward')
        ax.set_xlabel('Episode')
        plt.axhline(0, color='black')
        plt.axvline(0, color='black')
        fig.savefig('{}/reward_total.png'.format(self.path_summary))
        plt.draw()
        plt.close(fig)

    def close(self):
        self.writer.close()

    def launchTensorBoard(self):    
        os.system("tensorboard --logdir={} --port={}".format(
                                            self.path_summary,
                                            self.port))
        return


class Saver:
    """
        Class for saving and resuming the framework
    """ 

    def __init__(self, args):
        self.args = args

    def save_checkpoint(self,
                        save_dict,
                        episode,
                        filename = 'ckpt_last.pth.tar', 
                        is_best = False,
                        is_best_avg = False):
        """
            save on file
        """
        ckpt = self.build_state_ckpt(save_dict, episode)
        path_ckpt = os.path.join(self.args['dir_saved_models'], filename)
        K.save(ckpt, path_ckpt)

        if episode is not None:
            path_ckpt_ep = os.path.join(self.args['dir_saved_models'], 
                                        'ckpt_ep{}.pth.tar'.format(episode))
            shutil.copyfile(path_ckpt, path_ckpt_ep)

        if is_best:
            path_ckpt_best = os.path.join(self.args['dir_saved_models'], 
                                     'ckpt_best.pth.tar')
            shutil.copyfile(path_ckpt, path_ckpt_best)

        if is_best_avg:
            path_ckpt_best_avg = os.path.join(self.args['dir_saved_models'], 
                                     'ckpt_best_avg.pth.tar')
            shutil.copyfile(path_ckpt, path_ckpt_best_avg)


    def build_state_ckpt(self, save_dict, episode):
        """
            build a proper structure with all the info for resuming
        """
        ckpt = ({
            'args'       : self.args,
            'episode'    : episode,
            'save_dict'  : save_dict
            })

        return ckpt


    def resume_ckpt(self, resume_path=''):
        """
            build a proper structure with all the info for resuming
        """
        if resume_path =='':     
            ckpt = K.load(self.args['resume'])
        else:
            ckpt = K.load(resume_path)    

        self.args = ckpt['args']
        save_dict = ckpt['save_dict']
        episode   = ckpt['episode']

        return self.args, episode, save_dict
