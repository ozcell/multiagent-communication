import torch as K
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from macomm.agents import Actor, Critic
from macomm.exploration import gumbel_softmax
from macomm.environments import communication


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class MADDPG(object):
    
    def __init__(self, num_agents, observation_space, action_space, medium_space, optimizer, loss_func, gamma, tau, 
                 discrete=True, regularization=False, normalized_rewards=False, dtype=K.float32, device="cuda"):
        
        optimizer, lr = optimizer
        actor_lr, critic_lr = lr
        
        self.num_agents = num_agents
        self.loss_func = loss_func
        self.gamma = gamma
        self.tau = tau
        self.discrete = discrete
        self.regularization = regularization
        self.normalized_rewards = normalized_rewards
        self.dtype = dtype
        self.device = device
        self.has_communication = False
        
        # model initialization
        self.entities = []
        
        # actors
        self.actors = []
        self.actors_target = []
        self.actors_optim = []
        
        for i in range(num_agents):
            self.actors.append(Actor(observation_space, action_space, discrete).to(device))
            self.actors_target.append(Actor(observation_space, action_space, discrete).to(device))
            self.actors_optim.append(optimizer(self.actors[i].parameters(), lr = actor_lr))
            
        for i in range(num_agents):
            hard_update(self.actors_target[i], self.actors[i])

        self.entities.extend(self.actors)
        self.entities.extend(self.actors_target)
        self.entities.extend(self.actors_optim)

        # critics   
        self.critics = []
        self.critics_target = []
        self.critics_optim = []

        for i in range(num_agents):
            self.critics.append(Critic(observation_space*num_agents, action_space*num_agents).to(device))
            self.critics_target.append(Critic(observation_space*num_agents, action_space*num_agents).to(device))
            self.critics_optim.append(optimizer(self.critics[i].parameters(), lr = critic_lr))
                
        for i in range(num_agents):
            hard_update(self.critics_target[i], self.critics[i])

        self.entities.extend(self.critics)
        self.entities.extend(self.critics_target)
        self.entities.extend(self.critics_optim)

    def to_cpu(self):
        for entity in self.entities:
            if type(entity) != type(self.actors_optim[0]):
                entity.cpu()
        self.device = 'cpu'

    def to_cuda(self):        
        for entity in self.entities:
            if type(entity) != type(self.actors_optim[0]):
                entity.cuda()
        self.device = 'cuda'     

    def get_save_dict(self):
        """
        Save trained parameters of all agents into one file
        """
        # move parameters to CPU before saving
        self.to_cpu()
        save_dict = {'agent_params': [i.get_params() for i in self.agents]}
        return save_dict     
    
    def select_action(self, state, i_agent, exploration=False):
        with K.no_grad():
            mu = self.actors[i_agent](state.to(self.device))
        if self.discrete:
            mu = gumbel_softmax(mu, exploration=exploration)
        else:
            if exploration:
                mu += K.tensor(exploration.noise(), dtype=self.dtype, device=self.device)
            
        return mu.clamp(0, 1) 
                
    def update_parameters(self, batch, i_agent):
        
        mask = K.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=K.uint8, device=self.device)

        V = K.zeros((len(batch.state), 1), device=self.device)

        s = K.cat(batch.state, dim=1).to(self.device)
        a = K.cat(batch.action, dim=1).to(self.device)
        r = K.cat(batch.reward, dim=1).to(self.device)
        s_ = K.cat([i.to(self.device) for i in batch.next_state if i is not None], dim=1)
        a_ = K.zeros_like(a)[:,0:s_.shape[1],]

        if self.normalized_rewards:
            r -= r.mean()
            r /= r.std()

        Q = self.critics[i_agent](s, a)

        for i in range(self.num_agents):
            a_[i,] = gumbel_softmax(self.actors_target[i](s_[[i],]), exploration=False)

        V[mask] = self.critics_target[i_agent](s_, a_).detach()

        loss_critic = self.loss_func(Q, (V * self.gamma) + r[[i_agent],].squeeze(0)) 

        self.critics_optim[i_agent].zero_grad()
        loss_critic.backward()
        K.nn.utils.clip_grad_norm_(self.critics[i_agent].parameters(), 0.5)
        self.critics_optim[i_agent].step()

        for i in range(self.num_agents):
            a[i,] = gumbel_softmax(self.actors[i](s[[i],]), exploration=False)

        loss_actor = -self.critics[i_agent](s, a).mean()
        if self.regularization:
            loss_actor += (self.actors[i_agent].get_preactivations(s[[i_agent],])**2).mean()*1e-3

        self.actors_optim[i_agent].zero_grad()        
        loss_actor.backward()
        K.nn.utils.clip_grad_norm_(self.actors[i_agent].parameters(), 0.5)
        self.actors_optim[i_agent].step()

        soft_update(self.actors_target[i_agent], self.actors[i_agent], self.tau)
        soft_update(self.critics_target[i_agent], self.critics[i_agent], self.tau)
        
        return loss_critic.item(), loss_actor.item()      


class DDPG(MADDPG):
    ''' This is a version of MADDPG where there is no centralized training. 
    Each critic only observes its own observation, not the entire state.
    '''
    def __init__(self, num_agents, observation_space, action_space, medium_space, optimizer, loss_func, gamma, tau, 
                 discrete=True, regularization=False, normalized_rewards=False, dtype=K.float32, device="cuda"):
        
        super().__init__(num_agents, observation_space, action_space, medium_space, optimizer, loss_func, gamma, tau, 
                         discrete, regularization, normalized_rewards, dtype, device)

        optimizer, lr = optimizer
        _, critic_lr = lr

        # model initialization
        self.entities = []

        # actors
        self.entities.extend(self.actors)
        self.entities.extend(self.actors_target)
        self.entities.extend(self.actors_optim)
        
        # critics   
        self.critics = []
        self.critics_target = []
        self.critics_optim = []
        
        for i in range(num_agents):
            self.critics.append(Critic(observation_space, action_space).to(device))
            self.critics_target.append(Critic(observation_space, action_space).to(device))
            self.critics_optim.append(optimizer(self.critics[i].parameters(), lr = critic_lr))

        for i in range(num_agents):
            hard_update(self.critics_target[i], self.critics[i])

        self.entities.extend(self.critics)
        self.entities.extend(self.critics_target)
        self.entities.extend(self.critics_optim)

    def update_parameters(self, batch, i_agent):
        
        mask = K.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=K.uint8, device=self.device)

        V = K.zeros((len(batch.state), 1), device=self.device)

        s = K.cat(batch.state, dim=1).to(self.device)
        a = K.cat(batch.action, dim=1).to(self.device)
        r = K.cat(batch.reward, dim=1).to(self.device)
        s_ = K.cat([i.to(self.device) for i in batch.next_state if i is not None], dim=1)
        a_ = K.zeros_like(a)[:,0:s_.shape[1],]

        if self.normalized_rewards:
            r -= r.mean()
            r /= r.std()

        Q = self.critics[i_agent](s[[i_agent],], a[[i_agent],])

        for i in range(self.num_agents):
            a_[i,] = gumbel_softmax(self.actors_target[i](s_[[i],]), exploration=False)

        V[mask] = self.critics_target[i_agent](s_[[i_agent],], a_[[i_agent],]).detach()

        loss_critic = self.loss_func(Q, (V * self.gamma) + r[[i_agent],].squeeze(0)) 

        self.critics_optim[i_agent].zero_grad()
        loss_critic.backward()
        K.nn.utils.clip_grad_norm_(self.critics[i_agent].parameters(), 0.5)
        self.critics_optim[i_agent].step()

        for i in range(self.num_agents):
            a[i,] = gumbel_softmax(self.actors[i](s[[i],]), exploration=False)

        loss_actor = -self.critics[i_agent](s[[i_agent],], a[[i_agent],]).mean()
        if self.regularization:
            loss_actor += (self.actors[i_agent].get_preactivations(s[[i_agent],])**2).mean()*1e-3

        self.actors_optim[i_agent].zero_grad()        
        loss_actor.backward()
        K.nn.utils.clip_grad_norm_(self.actors[i_agent].parameters(), 0.5)
        self.actors_optim[i_agent].step()

        soft_update(self.actors_target[i_agent], self.actors[i_agent], self.tau)
        soft_update(self.critics_target[i_agent], self.critics[i_agent], self.tau)
        
        return loss_critic.item(), loss_actor.item()  


class MACDDPG(MADDPG):
    def __init__(self, num_agents, observation_space, action_space, medium_space, optimizer, loss_func, gamma, tau, 
                 discrete=True, regularization=False, normalized_rewards=False, dtype=K.float32, device="cuda"):
        
        super().__init__(num_agents, observation_space, action_space, medium_space, optimizer, loss_func, gamma, tau, 
                         discrete, regularization, normalized_rewards, dtype, device)

        optimizer, lr = optimizer
        actor_lr, critic_lr = lr

        self.num_agents = num_agents
        self.loss_func = loss_func
        self.gamma = gamma
        self.tau = tau
        self.discrete = discrete
        self.regularization = regularization
        self.normalized_rewards = normalized_rewards
        self.dtype = dtype
        self.device = device
        self.has_communication = True

        # model initialization
        self.entities = []

        # actors
        self.actors = []
        self.actors_target = []
        self.actors_optim = []
        
        for i in range(num_agents):
            self.actors.append(Actor(observation_space+medium_space, action_space, discrete).to(device))
            self.actors_target.append(Actor(observation_space+medium_space, action_space, discrete).to(device))
            self.actors_optim.append(optimizer(self.actors[i].parameters(), lr = actor_lr))
            
        for i in range(num_agents):
            hard_update(self.actors_target[i], self.actors[i])

        self.entities.extend(self.actors)
        self.entities.extend(self.actors_target)
        self.entities.extend(self.actors_optim) 
        
        # critics   
        self.critics = []
        self.critics_target = []
        self.critics_optim = []
        
        for i in range(num_agents):
            self.critics.append(Critic(observation_space+medium_space, action_space).to(device))
            self.critics_target.append(Critic(observation_space+medium_space, action_space).to(device))
            self.critics_optim.append(optimizer(self.critics[i].parameters(), lr = critic_lr))

        for i in range(num_agents):
            hard_update(self.critics_target[i], self.critics[i])
            
        self.entities.extend(self.critics)
        self.entities.extend(self.critics_target)
        self.entities.extend(self.critics_optim)    

        # communication actors
        self.comm_actors = []
        self.comm_actors_target = []
        self.comm_actors_optim = []
        
        for i in range(num_agents):
            self.comm_actors.append(Actor(observation_space, 1, discrete).to(device))
            self.comm_actors_target.append(Actor(observation_space, 1, discrete).to(device))
            self.comm_actors_optim.append(optimizer(self.comm_actors[i].parameters(), lr = actor_lr))
            
        for i in range(num_agents):
            hard_update(self.comm_actors_target[i], self.comm_actors[i])

        self.entities.extend(self.comm_actors)
        self.entities.extend(self.comm_actors_target)
        self.entities.extend(self.comm_actors_optim)

        # communication critics   
        self.comm_critics = []
        self.comm_critics_target = []
        self.comm_critics_optim = []
        
        for i in range(num_agents):
            self.comm_critics.append(Critic(observation_space*num_agents, 1*num_agents).to(device))
            self.comm_critics_target.append(Critic(observation_space*num_agents, 1*num_agents).to(device))
            self.comm_critics_optim.append(optimizer(self.comm_critics[i].parameters(), lr = critic_lr))

        for i in range(num_agents):
            hard_update(self.comm_critics_target[i], self.comm_critics[i])

        self.entities.extend(self.comm_critics)
        self.entities.extend(self.comm_critics_target)
        self.entities.extend(self.comm_critics_optim)
    
    def select_comm_action(self, state, i_agent, exploration=False):
        with K.no_grad():
            mu = self.comm_actors[i_agent](state.to(self.device))
            if exploration:
                mu += K.tensor(exploration.noise(), dtype=self.dtype, device=self.device)
        return mu.clamp(0, 1) 

    def update_parameters(self, batch, i_agent):
        
        mask = K.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=K.uint8, device=self.device)

        V = K.zeros((len(batch.state), 1), device=self.device)

        s = K.cat(batch.state, dim=1).to(self.device)
        a = K.cat(batch.action, dim=1).to(self.device)
        r = K.cat(batch.reward, dim=1).to(self.device)
        s_ = K.cat([i.to(self.device) for i in batch.next_state if i is not None], dim=1)
        a_ = K.zeros_like(a)[:,0:s_.shape[1],]
        
        W = K.zeros((len(batch.state), 1), device=self.device)
        
        m = K.cat(batch.medium, dim=1).to(self.device)
        t = K.cat(batch.comm_reward, dim=1).to(self.device)
        c = K.cat(batch.comm_action, dim=1).to(self.device)
        m_ = K.zeros_like(m)[:,0:s_.shape[1],]
        c_ = K.zeros_like(c)[:,0:s_.shape[1],]

        if self.normalized_rewards:
            r -= r.mean()
            r /= r.std()
            t -= t.mean()
            t /= t.std()

        R = self.comm_critics[i_agent](s, c)

        for i in range(self.num_agents):
            c_[i,] = self.comm_actors_target[i](s_[[i],])

        W[mask] = self.comm_critics_target[i_agent](s_, c_).detach()
        
        loss_comm_critic = self.loss_func(R, (W * self.gamma) + t[[i_agent],].squeeze(0) + r[[i_agent],].squeeze(0))

        self.comm_critics_optim[i_agent].zero_grad()
        loss_comm_critic.backward()
        K.nn.utils.clip_grad_norm_(self.comm_critics[i_agent].parameters(), 0.5)
        self.comm_critics_optim[i_agent].step()

        for i in range(self.num_agents):
            c[i,] = self.comm_actors[i](s[[i],])

        loss_comm_actor = -self.comm_critics[i_agent](s, c).mean()
        if self.regularization:
            loss_comm_actor += (self.comm_actors[i_agent].get_preactivations(s[[i_agent],])**2).mean()*1e-3

        self.comm_actors_optim[i_agent].zero_grad()        
        loss_comm_actor.backward()
        K.nn.utils.clip_grad_norm_(self.comm_actors[i_agent].parameters(), 0.5)
        self.comm_actors_optim[i_agent].step()

        Q = self.critics[i_agent](K.cat([s[[i_agent],], m], dim=-1),
                                  a[[i_agent],])

        m_ = communication().get_m(s_, c_)
        
        for i in range(self.num_agents):
            a_[i,] = gumbel_softmax(self.actors_target[i](K.cat([s_[[i],], m_], dim=-1)), exploration=False)

        V[mask] = self.critics_target[i_agent](K.cat([s_[[i_agent],], m_], dim=-1),
                                               a_[[i_agent],]).detach()

        loss_critic = self.loss_func(Q, (V * self.gamma) + r[[i_agent],].squeeze(0)) 

        self.critics_optim[i_agent].zero_grad()
        loss_critic.backward()
        K.nn.utils.clip_grad_norm_(self.critics[i_agent].parameters(), 0.5)
        self.critics_optim[i_agent].step()

        m = communication().get_m(s, c)

        for i in range(self.num_agents):
            a[i,] = gumbel_softmax(self.actors[i](K.cat([s[[i],], m], dim=-1)), exploration=False)

        loss_actor = -self.critics[i_agent](K.cat([s[[i_agent],], m], dim=-1), 
                                            a[[i_agent],]).mean()
        
        if self.regularization:
            loss_actor += (self.actors[i_agent].get_preactivations(K.cat([s[[i_agent],], m], dim=-1))**2).mean()*1e-3

        self.actors_optim[i_agent].zero_grad()        
        loss_actor.backward()
        K.nn.utils.clip_grad_norm_(self.actors[i_agent].parameters(), 0.5)
        self.actors_optim[i_agent].step()

        soft_update(self.comm_actors_target[i_agent], self.comm_actors[i_agent], self.tau)
        soft_update(self.comm_critics_target[i_agent], self.comm_critics[i_agent], self.tau)
        soft_update(self.actors_target[i_agent], self.actors[i_agent], self.tau)
        soft_update(self.critics_target[i_agent], self.critics[i_agent], self.tau)
        
        return loss_critic.item(), loss_actor.item()  
