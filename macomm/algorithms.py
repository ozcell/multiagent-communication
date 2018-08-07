import torch as K
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from macomm.exploration import gumbel_softmax

import pdb


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class MADDPG(object):
    
    def __init__(self, num_agents, observation_space, action_space, medium_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func=F.sigmoid,
                 discrete=True, regularization=False, normalized_rewards=False, communication=None, Comm_Actor=None, Comm_Critic=None, dtype=K.float32, device="cuda"):
        
        optimizer, lr = optimizer
        actor_lr, critic_lr = lr
        
        self.num_agents = num_agents
        self.loss_func = loss_func
        self.gamma = gamma
        self.tau = tau
        self.out_func = out_func
        self.discrete = discrete
        self.regularization = regularization
        self.normalized_rewards = normalized_rewards
        self.dtype = dtype
        self.device = device
        self.communication = communication
        self.action_space = action_space
        
        # model initialization
        self.entities = []
        
        # actors
        self.actors = []
        self.actors_target = []
        self.actors_optim = []
        
        for i in range(num_agents):
            self.actors.append(Actor(observation_space, action_space, discrete, out_func).to(device))
            self.actors_target.append(Actor(observation_space, action_space, discrete, out_func).to(device))
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
    
    def select_action(self, state, i_agent, exploration=False):
        self.actors[i_agent].eval()
        with K.no_grad():
            mu = self.actors[i_agent](state.to(self.device))
        self.actors[i_agent].train()
        if self.discrete:
            mu = gumbel_softmax(mu, exploration=exploration)
        else:
            if exploration:
                mu += K.tensor(exploration.noise(), dtype=self.dtype, device=self.device)
        
        if self.out_func == F.tanh:
            mu = mu.clamp(-1, 1)
        elif self.out_func == F.sigmoid:
            mu = mu.clamp(0, 1) 
        else:
            mu = mu

        return mu
                
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


class ORACLE(MADDPG):
    def __init__(self, num_agents, observation_space, action_space, medium_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func=F.sigmoid,
                 discrete=True, regularization=False, normalized_rewards=False, communication=None, Comm_Actor=None, Comm_Critic=None, dtype=K.float32, device="cuda"):

        super().__init__(num_agents, observation_space, action_space, medium_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func,
                         discrete, regularization, normalized_rewards, communication, Comm_Actor, Comm_Critic, dtype, device)

        optimizer, lr = optimizer
        actor_lr, _ = lr

        # model initialization
        self.entities = []

        # actors
        self.actors = []
        self.actors_target = []
        self.actors_optim = []
        
        for i in range(num_agents):
            self.actors.append(Actor(observation_space*num_agents, action_space, discrete, out_func).to(device))
            self.actors_target.append(Actor(observation_space*num_agents, action_space, discrete, out_func).to(device))
            self.actors_optim.append(optimizer(self.actors[i].parameters(), lr = actor_lr))
            
        for i in range(num_agents):
            hard_update(self.actors_target[i], self.actors[i])

        self.entities.extend(self.actors)
        self.entities.extend(self.actors_target)
        self.entities.extend(self.actors_optim)

        # critics   
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

        Q = self.critics[i_agent](s, a)

        for i in range(self.num_agents):
            a_[i,] = gumbel_softmax(self.actors_target[i](s_), exploration=False)

        V[mask] = self.critics_target[i_agent](s_, a_).detach()

        loss_critic = self.loss_func(Q, (V * self.gamma) + r[[i_agent],].squeeze(0)) 

        self.critics_optim[i_agent].zero_grad()
        loss_critic.backward()
        K.nn.utils.clip_grad_norm_(self.critics[i_agent].parameters(), 0.5)
        self.critics_optim[i_agent].step()

        for i in range(self.num_agents):
            a[i,] = gumbel_softmax(self.actors[i](s), exploration=False)

        loss_actor = -self.critics[i_agent](s, a).mean()
        if self.regularization:
            loss_actor += (self.actors[i_agent].get_preactivations(s)**2).mean()*1e-3

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
    def __init__(self, num_agents, observation_space, action_space, medium_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func=F.sigmoid,
                 discrete=True, regularization=False, normalized_rewards=False, communication=None, Comm_Actor=None, Comm_Critic=None, dtype=K.float32, device="cuda"):
        
        super().__init__(num_agents, observation_space, action_space, medium_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func,
                         discrete, regularization, normalized_rewards, communication, Comm_Actor, Comm_Critic, dtype, device)

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
    def __init__(self, num_agents, observation_space, action_space, medium_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func=F.sigmoid,
                 discrete=True, regularization=False, normalized_rewards=False, communication=None, Comm_Actor=None, Comm_Critic=None, dtype=K.float32, device="cuda"):
        
        super().__init__(num_agents, observation_space, action_space, medium_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func,
                         discrete, regularization, normalized_rewards, communication, Comm_Actor, Comm_Critic, dtype, device)

        optimizer, lr = optimizer
        actor_lr, critic_lr = lr

        #self.num_agents = num_agents
        #self.loss_func = loss_func
        #self.gamma = gamma
        #self.tau = tau
        #self.discrete = discrete
        #self.regularization = regularization
        #self.normalized_rewards = normalized_rewards
        #self.dtype = dtype
        #self.device = device
        #self.communication = communication

        # model initialization
        self.entities = []

        # actors
        self.actors = []
        self.actors_target = []
        self.actors_optim = []
        
        for i in range(num_agents):
            self.actors.append(Actor(observation_space+medium_space, action_space, discrete, out_func).to(device))
            self.actors_target.append(Actor(observation_space+medium_space, action_space, discrete, out_func).to(device))
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
            self.comm_actors.append(Comm_Actor(observation_space, 1, discrete, F.sigmoid).to(device))
            self.comm_actors_target.append(Comm_Actor(observation_space, 1, discrete, F.sigmoid).to(device))
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
            self.comm_critics.append(Comm_Critic(observation_space*num_agents, 1*num_agents).to(device))
            self.comm_critics_target.append(Comm_Critic(observation_space*num_agents, 1*num_agents).to(device))
            self.comm_critics_optim.append(optimizer(self.comm_critics[i].parameters(), lr = critic_lr))

        for i in range(num_agents):
            hard_update(self.comm_critics_target[i], self.comm_critics[i])

        self.entities.extend(self.comm_critics)
        self.entities.extend(self.comm_critics_target)
        self.entities.extend(self.comm_critics_optim)
    
    def select_comm_action(self, state, i_agent, exploration=False):
        self.comm_actors[i_agent].eval()
        with K.no_grad():
            mu = self.comm_actors[i_agent](state.to(self.device))
        self.comm_actors[i_agent].train()
        if exploration:
            mu += K.tensor(exploration.noise(), dtype=self.dtype, device=self.device)
        return mu.clamp(0, 1) 

    def update_parameters(self, batch, i_agent):
        
        mask = K.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=K.uint8, device=self.device)

        V = K.zeros((len(batch.state), 1), device=self.device)

        s = K.cat(batch.state, dim=1).to(self.device)
        a = K.cat(batch.action, dim=1).to(self.device)
        r = K.cat(batch.reward, dim=1).to(self.device)
        _a = K.cat(batch.prev_action, dim=1).to(self.device)
        s_ = K.cat([i.to(self.device) for i in batch.next_state if i is not None], dim=1)
        a_ = K.zeros_like(a)[:,0:s_.shape[1],]
        
        W = K.zeros((len(batch.state), 1), device=self.device)
        
        m = K.cat(batch.medium, dim=1).to(self.device)
        t = K.cat(batch.comm_reward, dim=1).to(self.device)
        c = K.cat(batch.comm_action, dim=1).to(self.device)
        m_ = K.zeros_like(m)[:,0:s_.shape[1],]
        c_ = K.zeros_like(c)[:,0:s_.shape[1],]

        if self.comm_actors[0].has_context:
            h = K.cat(batch.comm_context, dim=1).to(self.device)
            h_ = K.cat([i.to(self.device) for i in batch.next_comm_context if i is not None], dim=1)

        if self.normalized_rewards:
            r -= r.mean()
            r /= r.std()
            t -= t.mean()
            t /= t.std()

        R = self.comm_critics[i_agent](s, c)

        for i in range(self.num_agents):
            if self.comm_actors[0].has_context:
                c_[i,] = self.comm_actors_target[i](s_[[i],], h_[[i],])
            else:
                c_[i,] = self.comm_actors_target[i](s_[[i],])
        
        W[mask] = self.comm_critics_target[i_agent](s_, c_).detach()
        
        loss_comm_critic = self.loss_func(R, (W * self.gamma) + t[[i_agent],].squeeze(0) + r[[i_agent],].squeeze(0))
        #loss_comm_critic = self.loss_func(R, (W * self.gamma) + t[[i_agent],].squeeze(0))

        self.comm_critics_optim[i_agent].zero_grad()
        loss_comm_critic.backward()
        K.nn.utils.clip_grad_norm_(self.comm_critics[i_agent].parameters(), 0.5)
        self.comm_critics_optim[i_agent].step()

        for i in range(self.num_agents):
            if self.comm_actors[0].has_context:
                c[i,] = self.comm_actors[i](s[[i],], h[[i],])
            else:
                c[i,] = self.comm_actors[i](s[[i],])

        loss_comm_actor = -self.comm_critics[i_agent](s, c).mean()
        if self.regularization:
            if self.comm_actors[0].has_context:
                loss_comm_actor += (self.comm_actors[i_agent].get_preactivations(s[[i_agent],], h[[i_agent],])**2).mean()*1e-3
            else:
                loss_comm_actor += (self.comm_actors[i_agent].get_preactivations(s[[i_agent],])**2).mean()*1e-3

        self.comm_actors_optim[i_agent].zero_grad()        
        loss_comm_actor.backward()
        K.nn.utils.clip_grad_norm_(self.comm_actors[i_agent].parameters(), 0.5)
        self.comm_actors_optim[i_agent].step()

        Q = self.critics[i_agent](K.cat([s[[i_agent],], m], dim=-1),
                                  a[[i_agent],])

        m_ = self.communication.get_m(s_, c_, a[:, mask, :])
        
        for i in range(self.num_agents):
            a_[i,] = gumbel_softmax(self.actors_target[i](K.cat([s_[[i],], m_], dim=-1)), exploration=False)

        V[mask] = self.critics_target[i_agent](K.cat([s_[[i_agent],], m_], dim=-1),
                                               a_[[i_agent],]).detach()

        loss_critic = self.loss_func(Q, (V * self.gamma) + r[[i_agent],].squeeze(0)) 

        self.critics_optim[i_agent].zero_grad()
        loss_critic.backward()
        K.nn.utils.clip_grad_norm_(self.critics[i_agent].parameters(), 0.5)
        self.critics_optim[i_agent].step()

        m = self.communication.get_m(s, c, _a)

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

    
class MACCDDPG(MACDDPG): 
    def __init__(self, num_agents, observation_space, action_space, medium_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func=F.sigmoid,
                 discrete=True, regularization=False, normalized_rewards=False, communication=None, Comm_Actor=None, Comm_Critic=None, dtype=K.float32, device="cuda"):
        
        super().__init__(num_agents, observation_space, action_space, medium_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func,
                         discrete, regularization, normalized_rewards, communication, Comm_Actor, Comm_Critic, dtype, device)

        optimizer, lr = optimizer
        actor_lr, critic_lr = lr

        # model initialization
        self.entities = []

        # actors
        self.actors = []
        self.actors_target = []
        self.actors_optim = []
        
        for i in range(num_agents):
            self.actors.append(Actor(observation_space+medium_space, action_space, discrete, out_func).to(device))
            self.actors_target.append(Actor(observation_space+medium_space, action_space, discrete, out_func).to(device))
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

        # communication actors
        self.comm_actors = []
        self.comm_actors_target = []
        self.comm_actors_optim = []
        
        for i in range(num_agents):
            self.comm_actors.append(Comm_Actor(observation_space, 1, discrete, F.sigmoid).to(device))
            self.comm_actors_target.append(Comm_Actor(observation_space, 1, discrete, F.sigmoid).to(device))
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
            self.comm_critics.append(Comm_Critic(observation_space*num_agents, 1*num_agents).to(device))
            self.comm_critics_target.append(Comm_Critic(observation_space*num_agents, 1*num_agents).to(device))
            self.comm_critics_optim.append(optimizer(self.comm_critics[i].parameters(), lr = critic_lr))

        for i in range(num_agents):
            hard_update(self.comm_critics_target[i], self.comm_critics[i])

        self.entities.extend(self.comm_critics)
        self.entities.extend(self.comm_critics_target)
        self.entities.extend(self.comm_critics_optim)

    def update_parameters(self, batch, i_agent):
        
        mask = K.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=K.uint8, device=self.device)

        V = K.zeros((len(batch.state), 1), device=self.device)

        s = K.cat(batch.state, dim=1).to(self.device)
        a = K.cat(batch.action, dim=1).to(self.device)
        r = K.cat(batch.reward, dim=1).to(self.device)
        _a = K.cat(batch.prev_action, dim=1).to(self.device)
        s_ = K.cat([i.to(self.device) for i in batch.next_state if i is not None], dim=1)
        a_ = K.zeros_like(a)[:,0:s_.shape[1],]
        
        W = K.zeros((len(batch.state), 1), device=self.device)
        
        m = K.cat(batch.medium, dim=1).to(self.device)
        t = K.cat(batch.comm_reward, dim=1).to(self.device)
        c = K.cat(batch.comm_action, dim=1).to(self.device)
        m_ = K.zeros_like(m)[:,0:s_.shape[1],]
        c_ = K.zeros_like(c)[:,0:s_.shape[1],]

        if self.comm_actors[0].has_context:
            h = K.cat(batch.comm_context, dim=1).to(self.device)
            h_ = K.cat([i.to(self.device) for i in batch.next_comm_context if i is not None], dim=1)

        if self.normalized_rewards:
            r -= r.mean()
            r /= r.std()
            t -= t.mean()
            t /= t.std()

        R = self.comm_critics[i_agent](s, c)

        for i in range(self.num_agents):
            if self.comm_actors[0].has_context:
                c_[i,] = self.comm_actors_target[i](s_[[i],], h_[[i],])
            else:
                c_[i,] = self.comm_actors_target[i](s_[[i],])
        
        W[mask] = self.comm_critics_target[i_agent](s_, c_).detach()
        
        loss_comm_critic = self.loss_func(R, (W * self.gamma) + t[[i_agent],].squeeze(0) + r[[i_agent],].squeeze(0))
        #loss_comm_critic = self.loss_func(R, (W * self.gamma) + t[[i_agent],].squeeze(0))

        self.comm_critics_optim[i_agent].zero_grad()
        loss_comm_critic.backward()
        K.nn.utils.clip_grad_norm_(self.comm_critics[i_agent].parameters(), 0.5)
        self.comm_critics_optim[i_agent].step()

        for i in range(self.num_agents):
            if self.comm_actors[0].has_context:
                c[i,] = self.comm_actors[i](s[[i],], h[[i],])
            else:
                c[i,] = self.comm_actors[i](s[[i],])

        loss_comm_actor = -self.comm_critics[i_agent](s, c).mean()
        if self.regularization:
            if self.comm_actors[0].has_context:
                loss_comm_actor += (self.comm_actors[i_agent].get_preactivations(s[[i_agent],], h[[i_agent],])**2).mean()*1e-3
            else:
                loss_comm_actor += (self.comm_actors[i_agent].get_preactivations(s[[i_agent],])**2).mean()*1e-3

        self.comm_actors_optim[i_agent].zero_grad()        
        loss_comm_actor.backward()
        K.nn.utils.clip_grad_norm_(self.comm_actors[i_agent].parameters(), 0.5)
        self.comm_actors_optim[i_agent].step()

        Q = self.critics[i_agent](s, a)

        m_ = self.communication.get_m(s_, c_, a[:, mask, :])
        
        for i in range(self.num_agents):
            a_[i,] = gumbel_softmax(self.actors_target[i](K.cat([s_[[i],], m_], dim=-1)), exploration=False)

        V[mask] = self.critics_target[i_agent](s_, a_).detach()

        loss_critic = self.loss_func(Q, (V * self.gamma) + r[[i_agent],].squeeze(0)) 

        self.critics_optim[i_agent].zero_grad()
        loss_critic.backward()
        K.nn.utils.clip_grad_norm_(self.critics[i_agent].parameters(), 0.5)
        self.critics_optim[i_agent].step()

        m = self.communication.get_m(s, c, _a)

        for i in range(self.num_agents):
            a[i,] = gumbel_softmax(self.actors[i](K.cat([s[[i],], m], dim=-1)), exploration=False)

        loss_actor = -self.critics[i_agent](s, a).mean()
        
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


class MADCDDPG(MACDDPG):
    def __init__(self, num_agents, observation_space, action_space, medium_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func=F.sigmoid,
                 discrete=True, regularization=False, normalized_rewards=False, communication=None, Comm_Actor=None, Comm_Critic=None, dtype=K.float32, device="cuda"):
        
        super().__init__(num_agents, observation_space, action_space, medium_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func,
                         discrete, regularization, normalized_rewards, communication, Comm_Actor, Comm_Critic, dtype, device)

        optimizer, lr = optimizer
        actor_lr, critic_lr = lr

        #self.num_agents = num_agents
        #self.loss_func = loss_func
        #self.gamma = gamma
        #self.tau = tau
        #self.discrete = discrete
        #self.regularization = regularization
        #self.normalized_rewards = normalized_rewards
        #self.dtype = dtype
        #self.device = device
        #self.communication = communication

        # model initialization
        self.entities = []

        # actors
        self.actors = []
        self.actors_target = []
        self.actors_optim = []
        
        for i in range(num_agents):
            self.actors.append(Actor(observation_space+medium_space, action_space, discrete, out_func).to(device))
            self.actors_target.append(Actor(observation_space+medium_space, action_space, discrete, out_func).to(device))
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
            self.comm_actors.append(Comm_Actor(observation_space, num_agents, discrete, F.softmax).to(device))
            self.comm_actors_target.append(Comm_Actor(observation_space, num_agents, discrete, F.softmax).to(device))
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
            self.comm_critics.append(Comm_Critic(observation_space, num_agents).to(device))
            self.comm_critics_target.append(Comm_Critic(observation_space, num_agents).to(device))
            self.comm_critics_optim.append(optimizer(self.comm_critics[i].parameters(), lr = critic_lr))

        for i in range(num_agents):
            hard_update(self.comm_critics_target[i], self.comm_critics[i])

        self.entities.extend(self.comm_critics)
        self.entities.extend(self.comm_critics_target)
        self.entities.extend(self.comm_critics_optim)
    
    def select_comm_action(self, state, i_agent, exploration=False):
        self.comm_actors[i_agent].eval()
        with K.no_grad():
            mu = self.comm_actors[i_agent](state.to(self.device))
        self.comm_actors[i_agent].train()
        if exploration:
            mu += K.tensor(exploration.noise(), dtype=self.dtype, device=self.device)
        return mu.clamp(0, 1) 

    def update_parameters(self, batch, i_agent):
        
        mask = K.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=K.uint8, device=self.device)

        V = K.zeros((len(batch.state), 1), device=self.device)

        s = K.cat(batch.state, dim=1).to(self.device)
        a = K.cat(batch.action, dim=1).to(self.device)
        r = K.cat(batch.reward, dim=1).to(self.device)
        _a = K.cat(batch.prev_action, dim=1).to(self.device)
        s_ = K.cat([i.to(self.device) for i in batch.next_state if i is not None], dim=1)
        a_ = K.zeros_like(a)[:,0:s_.shape[1],]
        
        W = K.zeros((len(batch.state), 1), device=self.device)
        
        m = K.cat(batch.medium, dim=1).to(self.device)
        t = K.cat(batch.comm_reward, dim=1).to(self.device)
        c = K.cat(batch.comm_action, dim=1).to(self.device)
        m_ = K.zeros_like(m)[:,0:s_.shape[1],]
        c_ = K.zeros_like(c)[:,0:s_.shape[1],]

        if self.comm_actors[0].has_context:
            h = K.cat(batch.comm_context, dim=1).to(self.device)
            h_ = K.cat([i.to(self.device) for i in batch.next_comm_context if i is not None], dim=1)

        if self.normalized_rewards:
            r -= r.mean()
            r /= r.std()
            t -= t.mean()
            t /= t.std()

        R = self.comm_critics[i_agent](s[[i_agent],],
                                       c[[i_agent],])

        for i in range(self.num_agents):
            if self.comm_actors[0].has_context:
                c_[i,] = self.comm_actors_target[i](s_[[i],], h_[[i],])
            else:
                c_[i,] = self.comm_actors_target[i](s_[[i],])
        
        W[mask] = self.comm_critics_target[i_agent](s_[[i_agent],], 
                                                    c_[[i_agent],]).detach()

        #foo1 = t[[i_agent],].squeeze(0)
        #foo2 = r[[i_agent],].squeeze(0)

        #foo1 = (foo1-foo1.min())/(foo1.max()-foo1.min())
        #foo2 = (foo2-foo2.min())/(foo2.max()-foo2.min())
        
        #loss_comm_critic = self.loss_func(R, (W * self.gamma) + foo1 + foo2)
        loss_comm_critic = self.loss_func(R, (W * self.gamma) + t[[i_agent],].squeeze(0) + r[[i_agent],].squeeze(0))
        #loss_comm_critic = self.loss_func(R, (W * self.gamma) + t[[i_agent],].squeeze(0))

        self.comm_critics_optim[i_agent].zero_grad()
        loss_comm_critic.backward()
        K.nn.utils.clip_grad_norm_(self.comm_critics[i_agent].parameters(), 0.5)
        self.comm_critics_optim[i_agent].step()

        for i in range(self.num_agents):
            if self.comm_actors[0].has_context:
                c[i,] = self.comm_actors[i](s[[i],], h[[i],])
            else:
                c[i,] = self.comm_actors[i](s[[i],])

        loss_comm_actor = -self.comm_critics[i_agent](s[[i_agent],], 
                                                      c[[i_agent],]).mean()
        if self.regularization:
            if self.comm_actors[0].has_context:
                loss_comm_actor += (self.comm_actors[i_agent].get_preactivations(s[[i_agent],], h[[i_agent],])**2).mean()*1e-3
            else:
                loss_comm_actor += (self.comm_actors[i_agent].get_preactivations(s[[i_agent],])**2).mean()*1e-3

        self.comm_actors_optim[i_agent].zero_grad()        
        loss_comm_actor.backward()
        K.nn.utils.clip_grad_norm_(self.comm_actors[i_agent].parameters(), 0.5)
        self.comm_actors_optim[i_agent].step()

        Q = self.critics[i_agent](K.cat([s[[i_agent],], m[[i_agent],]], dim=-1),
                                  a[[i_agent],])

        m_ = self.communication.get_m(s_, c_, a[:, mask, :])
        
        for i in range(self.num_agents):
            a_[i,] = gumbel_softmax(self.actors_target[i](K.cat([s_[[i],], m_[[i],]], dim=-1)), exploration=False)

        V[mask] = self.critics_target[i_agent](K.cat([s_[[i_agent],], m_[[i_agent],]], dim=-1),
                                               a_[[i_agent],]).detach()

        loss_critic = self.loss_func(Q, (V * self.gamma) + r[[i_agent],].squeeze(0)) 

        self.critics_optim[i_agent].zero_grad()
        loss_critic.backward()
        K.nn.utils.clip_grad_norm_(self.critics[i_agent].parameters(), 0.5)
        self.critics_optim[i_agent].step()

        m = self.communication.get_m(s, c, _a)

        for i in range(self.num_agents):
            a[i,] = gumbel_softmax(self.actors[i](K.cat([s[[i],], m[[i],]], dim=-1)), exploration=False)

        loss_actor = -self.critics[i_agent](K.cat([s[[i_agent],], m[[i_agent],]], dim=-1), 
                                            a[[i_agent],]).mean()
        
        if self.regularization:
            loss_actor += (self.actors[i_agent].get_preactivations(K.cat([s[[i_agent],], m[[i_agent],]], dim=-1))**2).mean()*1e-3

        self.actors_optim[i_agent].zero_grad()        
        loss_actor.backward()
        K.nn.utils.clip_grad_norm_(self.actors[i_agent].parameters(), 0.5)
        self.actors_optim[i_agent].step()

        soft_update(self.comm_actors_target[i_agent], self.comm_actors[i_agent], self.tau)
        soft_update(self.comm_critics_target[i_agent], self.comm_critics[i_agent], self.tau)
        soft_update(self.actors_target[i_agent], self.actors[i_agent], self.tau)
        soft_update(self.critics_target[i_agent], self.critics[i_agent], self.tau)
        
        return loss_critic.item(), loss_actor.item()


class MSDDPG(MADCDDPG):
    def __init__(self, num_agents, observation_space, action_space, medium_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func=F.sigmoid,
                 discrete=True, regularization=False, normalized_rewards=False, communication=None, Comm_Actor=None, Comm_Critic=None, dtype=K.float32, device="cuda"):
        
        super().__init__(num_agents, observation_space, action_space, medium_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func,
                         discrete, regularization, normalized_rewards, communication, Comm_Actor, Comm_Critic, dtype, device)

        optimizer, lr = optimizer
        actor_lr, critic_lr = lr

        #self.num_agents = num_agents
        #self.loss_func = loss_func
        #self.gamma = gamma
        #self.tau = tau
        #self.discrete = discrete
        #self.regularization = regularization
        #self.normalized_rewards = normalized_rewards
        #self.dtype = dtype
        #self.device = device
        #self.communication = communication

        # model initialization
        self.entities = []

        # actors
        self.actors = []
        self.actors_target = []
        self.actors_optim = []
        
        for i in range(num_agents):
            self.actors.append(Actor(observation_space+medium_space, action_space, discrete, out_func).to(device))
            self.actors_target.append(Actor(observation_space+medium_space, action_space, discrete, out_func).to(device))
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

    def update_parameters(self, batch, i_agent):
        
        mask = K.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=K.uint8, device=self.device)

        V = K.zeros((len(batch.state), 1), device=self.device)

        s = K.cat(batch.state, dim=1).to(self.device)
        a = K.cat(batch.action, dim=1).to(self.device)
        r = K.cat(batch.reward, dim=1).to(self.device)
        s_ = K.cat([i.to(self.device) for i in batch.next_state if i is not None], dim=1)
        a_ = K.zeros_like(a)[:,0:s_.shape[1],]

        m = s[[0],]
        m_ = s_[[0],]
        
        if self.normalized_rewards:
            r -= r.mean()
            r /= r.std()

        Q = self.critics[i_agent](K.cat([s[[i_agent],], m], dim=-1),
                                  a[[i_agent],])
        
        for i in range(self.num_agents):
            a_[i,] = gumbel_softmax(self.actors_target[i](K.cat([s_[[i],], m_], dim=-1)), exploration=False)

        V[mask] = self.critics_target[i_agent](K.cat([s_[[i_agent],], m_], dim=-1),
                                               a_[[i_agent],]).detach()

        loss_critic = self.loss_func(Q, (V * self.gamma) + r[[i_agent],].squeeze(0)) 

        self.critics_optim[i_agent].zero_grad()
        loss_critic.backward()
        K.nn.utils.clip_grad_norm_(self.critics[i_agent].parameters(), 0.5)
        self.critics_optim[i_agent].step()

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


class MS3DDPG(MADCDDPG):
    def __init__(self, num_agents, observation_space, action_space, medium_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func=F.sigmoid,
                 discrete=True, regularization=False, normalized_rewards=False, communication=None, Comm_Actor=None, Comm_Critic=None, dtype=K.float32, device="cuda"):
        
        super().__init__(num_agents, observation_space, action_space, medium_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func,
                         discrete, regularization, normalized_rewards, communication, Comm_Actor, Comm_Critic, dtype, device)

        optimizer, lr = optimizer
        actor_lr, critic_lr = lr

        #self.num_agents = num_agents
        #self.loss_func = loss_func
        #self.gamma = gamma
        #self.tau = tau
        #self.discrete = discrete
        #self.regularization = regularization
        #self.normalized_rewards = normalized_rewards
        #self.dtype = dtype
        #self.device = device
        #self.communication = communication

        # model initialization
        self.entities = []

        # actors
        self.actors = []
        self.actors_target = []
        self.actors_optim = []
        
        for i in range(num_agents):
            self.actors.append(Actor(observation_space+medium_space, action_space, discrete, out_func).to(device))
            self.actors_target.append(Actor(observation_space+medium_space, action_space, discrete, out_func).to(device))
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

    def update_parameters(self, batch, i_agent):
        
        mask = K.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=K.uint8, device=self.device)

        V = K.zeros((len(batch.state), 1), device=self.device)

        s = K.cat(batch.state, dim=1).to(self.device)
        a = K.cat(batch.action, dim=1).to(self.device)
        r = K.cat(batch.reward, dim=1).to(self.device)
        s_ = K.cat([i.to(self.device) for i in batch.next_state if i is not None], dim=1)
        a_ = K.zeros_like(a)[:,0:s_.shape[1],]

        m = s[[(i_agent-1)%self.num_agents],]
        m_ = s_[[(i_agent-1)%self.num_agents],]
        
        if self.normalized_rewards:
            r -= r.mean()
            r /= r.std()

        Q = self.critics[i_agent](K.cat([s[[i_agent],], m], dim=-1),
                                  a[[i_agent],])
        
        for i in range(self.num_agents):
            a_[i,] = gumbel_softmax(self.actors_target[i](K.cat([s_[[i],], m_], dim=-1)), exploration=False)

        V[mask] = self.critics_target[i_agent](K.cat([s_[[i_agent],], m_], dim=-1),
                                               a_[[i_agent],]).detach()

        loss_critic = self.loss_func(Q, (V * self.gamma) + r[[i_agent],].squeeze(0)) 

        self.critics_optim[i_agent].zero_grad()
        loss_critic.backward()
        K.nn.utils.clip_grad_norm_(self.critics[i_agent].parameters(), 0.5)
        self.critics_optim[i_agent].step()

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


class MADCDDPG_WS(MACDDPG):
    def __init__(self, num_agents, observation_space, action_space, medium_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func=F.sigmoid,
                 discrete=True, regularization=False, normalized_rewards=False, communication=None, Comm_Actor=None, Comm_Critic=None, dtype=K.float32, device="cuda"):
        
        super().__init__(num_agents, observation_space, action_space, medium_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func,
                         discrete, regularization, normalized_rewards, communication, Comm_Actor, Comm_Critic, dtype, device)

        optimizer, lr = optimizer
        actor_lr, critic_lr = lr

        #self.num_agents = num_agents
        #self.loss_func = loss_func
        #self.gamma = gamma
        #self.tau = tau
        #self.discrete = discrete
        #self.regularization = regularization
        #self.normalized_rewards = normalized_rewards
        #self.dtype = dtype
        #self.device = device
        #self.communication = communication

        # model initialization
        self.entities = []

        # actors
        self.actors = []
        self.actors_target = []
        self.actors_optim = []
        
        for i in range(num_agents):
            self.actors.append(Actor(observation_space+medium_space, action_space, discrete, out_func).to(device))
            self.actors_target.append(Actor(observation_space+medium_space, action_space, discrete, out_func).to(device))
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
        
        for i in range(1):
            self.critics.append(Critic(observation_space+medium_space+1, action_space).to(device))
            self.critics_target.append(Critic(observation_space+medium_space+1, action_space).to(device))
            self.critics_optim.append(optimizer(self.critics[i].parameters(), lr = critic_lr))

        for i in range(1):
            hard_update(self.critics_target[i], self.critics[i])
            
        self.entities.extend(self.critics)
        self.entities.extend(self.critics_target)
        self.entities.extend(self.critics_optim)    

        # communication actors
        self.comm_actors = []
        self.comm_actors_target = []
        self.comm_actors_optim = []
        
        for i in range(num_agents):
            self.comm_actors.append(Comm_Actor(observation_space, 1, discrete, F.sigmoid).to(device))
            self.comm_actors_target.append(Comm_Actor(observation_space, 1, discrete, F.sigmoid).to(device))
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
        
        for i in range(1):
            self.comm_critics.append(Comm_Critic(observation_space+1, 1).to(device))
            self.comm_critics_target.append(Comm_Critic(observation_space+1, 1).to(device))
            self.comm_critics_optim.append(optimizer(self.comm_critics[i].parameters(), lr = critic_lr))

        for i in range(1):
            hard_update(self.comm_critics_target[i], self.comm_critics[i])

        self.entities.extend(self.comm_critics)
        self.entities.extend(self.comm_critics_target)
        self.entities.extend(self.comm_critics_optim)
    
    def select_comm_action(self, state, i_agent, exploration=False):
        self.comm_actors[i_agent].eval()
        with K.no_grad():
            mu = self.comm_actors[i_agent](state.to(self.device))
        self.comm_actors[i_agent].train()
        if exploration:
            mu += K.tensor(exploration.noise(), dtype=self.dtype, device=self.device)
        return mu.clamp(0, 1) 

    def update_parameters(self, batch, i_agent):
        
        mask = K.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=K.uint8, device=self.device)
        

        V = K.zeros((len(batch.state), 1), device=self.device)

        s = K.cat(batch.state, dim=1).to(self.device)
        a = K.cat(batch.action, dim=1).to(self.device)
        r = K.cat(batch.reward, dim=1).to(self.device)
        _a = K.cat(batch.prev_action, dim=1).to(self.device)
        s_ = K.cat([i.to(self.device) for i in batch.next_state if i is not None], dim=1)
        a_ = K.zeros_like(a)[:,0:s_.shape[1],]
        
        W = K.zeros((len(batch.state), 1), device=self.device)
        
        m = K.cat(batch.medium, dim=1).to(self.device)
        t = K.cat(batch.comm_reward, dim=1).to(self.device)
        c = K.cat(batch.comm_action, dim=1).to(self.device)
        m_ = K.zeros_like(m)[:,0:s_.shape[1],]
        c_ = K.zeros_like(c)[:,0:s_.shape[1],]

        if self.comm_actors[0].has_context:
            h = K.cat(batch.comm_context, dim=1).to(self.device)
            h_ = K.cat([i.to(self.device) for i in batch.next_comm_context if i is not None], dim=1)

        if self.normalized_rewards:
            r -= r.mean()
            r /= r.std()
            t -= t.mean()
            t /= t.std()

        R = self.comm_critics[0](K.cat([s[[i_agent],], i_agent*K.ones((1,s.shape[1],1), dtype=self.dtype, device=self.device)], dim=-1),
                                       c[[i_agent],])

        for i in range(self.num_agents):
            if self.comm_actors[0].has_context:
                c_[i,] = self.comm_actors_target[i](s_[[i],], h_[[i],])
            else:
                c_[i,] = self.comm_actors_target[i](s_[[i],])
        
        W[mask] = self.comm_critics_target[0](K.cat([s_[[i_agent],], i_agent*K.ones((1,s_.shape[1],1), dtype=self.dtype, device=self.device)], dim=-1),
                                                    c_[[i_agent],]).detach()

        #foo1 = t[[i_agent],].squeeze(0)
        #foo2 = r[[i_agent],].squeeze(0)

        #foo1 = (foo1-foo1.min())/(foo1.max()-foo1.min())
        #foo2 = (foo2-foo2.min())/(foo2.max()-foo2.min())
        
        #loss_comm_critic = self.loss_func(R, (W * self.gamma) + foo1 + foo2)
        loss_comm_critic = self.loss_func(R, (W * self.gamma) + t[[i_agent],].squeeze(0) + r[[i_agent],].squeeze(0))
        #loss_comm_critic = self.loss_func(R, (W * self.gamma) + t[[i_agent],].squeeze(0))

        self.comm_critics_optim[0].zero_grad()
        loss_comm_critic.backward()
        K.nn.utils.clip_grad_norm_(self.comm_critics[0].parameters(), 0.5)
        self.comm_critics_optim[0].step()

        for i in range(self.num_agents):
            if self.comm_actors[0].has_context:
                c[i,] = self.comm_actors[i](s[[i],], h[[i],])
            else:
                c[i,] = self.comm_actors[i](s[[i],])

        loss_comm_actor = -self.comm_critics[0](K.cat([s[[i_agent],], i_agent*K.ones((1,s.shape[1],1), dtype=self.dtype, device=self.device)], dim=-1),
                                                      c[[i_agent],]).mean()
        if self.regularization:
            if self.comm_actors[0].has_context:
                loss_comm_actor += (self.comm_actors[i_agent].get_preactivations(s[[i_agent],], h[[i_agent],])**2).mean()*1e-3
            else:
                loss_comm_actor += (self.comm_actors[i_agent].get_preactivations(s[[i_agent],])**2).mean()*1e-3

        self.comm_actors_optim[i_agent].zero_grad()        
        loss_comm_actor.backward()
        K.nn.utils.clip_grad_norm_(self.comm_actors[i_agent].parameters(), 0.5)
        self.comm_actors_optim[i_agent].step()

        Q = self.critics[0](K.cat([s[[i_agent],], i_agent*K.ones((1,s.shape[1],1), dtype=self.dtype, device=self.device), m], dim=-1),
                                  a[[i_agent],])

        m_ = self.communication.get_m(s_, c_, a[:, mask, :])
        
        for i in range(self.num_agents):
            a_[i,] = gumbel_softmax(self.actors_target[i](K.cat([s_[[i],], m_], dim=-1)), exploration=False)

        V[mask] = self.critics_target[0](K.cat([s_[[i_agent],], i_agent*K.ones((1,s_.shape[1],1), dtype=self.dtype, device=self.device), m_], dim=-1),
                                               a_[[i_agent],]).detach()

        loss_critic = self.loss_func(Q, (V * self.gamma) + r[[i_agent],].squeeze(0)) 

        self.critics_optim[0].zero_grad()
        loss_critic.backward()
        K.nn.utils.clip_grad_norm_(self.critics[0].parameters(), 0.5)
        self.critics_optim[0].step()

        m = self.communication.get_m(s, c, _a)

        for i in range(self.num_agents):
            a[i,] = gumbel_softmax(self.actors[i](K.cat([s[[i],], m], dim=-1)), exploration=False)

        loss_actor = -self.critics[0](K.cat([s[[i_agent],], i_agent*K.ones((1,s.shape[1],1), dtype=self.dtype, device=self.device), m], dim=-1), 
                                            a[[i_agent],]).mean()
        
        if self.regularization:
            loss_actor += (self.actors[i_agent].get_preactivations(K.cat([s[[i_agent],], m], dim=-1))**2).mean()*1e-3

        self.actors_optim[i_agent].zero_grad()        
        loss_actor.backward()
        K.nn.utils.clip_grad_norm_(self.actors[i_agent].parameters(), 0.5)
        self.actors_optim[i_agent].step()

        soft_update(self.comm_actors_target[i_agent], self.comm_actors[i_agent], self.tau)
        soft_update(self.comm_critics_target[0], self.comm_critics[0], self.tau)
        soft_update(self.actors_target[i_agent], self.actors[i_agent], self.tau)
        soft_update(self.critics_target[0], self.critics[0], self.tau)
        
        return loss_critic.item(), loss_actor.item()


class MADCDDPG_WSC(MACDDPG):
    def __init__(self, num_agents, observation_space, action_space, medium_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func=F.sigmoid,
                 discrete=True, regularization=False, normalized_rewards=False, communication=None, Comm_Actor=None, Comm_Critic=None, dtype=K.float32, device="cuda"):
        
        super().__init__(num_agents, observation_space, action_space, medium_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func,
                         discrete, regularization, normalized_rewards, communication, Comm_Actor, Comm_Critic, dtype, device)

        optimizer, lr = optimizer
        actor_lr, critic_lr = lr

        #self.num_agents = num_agents
        #self.loss_func = loss_func
        #self.gamma = gamma
        #self.tau = tau
        #self.discrete = discrete
        #self.regularization = regularization
        #self.normalized_rewards = normalized_rewards
        #self.dtype = dtype
        #self.device = device
        #self.communication = communication

        # model initialization
        self.entities = []

        # actors
        self.actors = []
        self.actors_target = []
        self.actors_optim = []
        
        for i in range(num_agents):
            self.actors.append(Actor(observation_space+medium_space, action_space, discrete, out_func).to(device))
            self.actors_target.append(Actor(observation_space+medium_space, action_space, discrete, out_func).to(device))
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
        
        for i in range(1):
            self.comm_actors.append(Comm_Actor(observation_space+1, 1, discrete, F.sigmoid).to(device))
            self.comm_actors_target.append(Comm_Actor(observation_space+1, 1, discrete, F.sigmoid).to(device))
            self.comm_actors_optim.append(optimizer(self.comm_actors[i].parameters(), lr = actor_lr))
            
        for i in range(1):
            hard_update(self.comm_actors_target[i], self.comm_actors[i])

        self.entities.extend(self.comm_actors)
        self.entities.extend(self.comm_actors_target)
        self.entities.extend(self.comm_actors_optim)

        # communication critics   
        self.comm_critics = []
        self.comm_critics_target = []
        self.comm_critics_optim = []
        
        for i in range(1):
            self.comm_critics.append(Comm_Critic(observation_space+1, 1).to(device))
            self.comm_critics_target.append(Comm_Critic(observation_space+1, 1).to(device))
            self.comm_critics_optim.append(optimizer(self.comm_critics[i].parameters(), lr = critic_lr))

        for i in range(1):
            hard_update(self.comm_critics_target[i], self.comm_critics[i])

        self.entities.extend(self.comm_critics)
        self.entities.extend(self.comm_critics_target)
        self.entities.extend(self.comm_critics_optim)
    
    def select_comm_action(self, state, i_agent, exploration=False):
        self.comm_actors[0].eval()
        with K.no_grad():
            mu = self.comm_actors[0](K.cat([state.to(self.device), i_agent*K.ones((1,state.shape[1],1), dtype=self.dtype, device=self.device)], dim=-1))
        self.comm_actors[0].train()
        if exploration:
            mu += K.tensor(exploration.noise(), dtype=self.dtype, device=self.device)
        return mu.clamp(0, 1) 

    def update_parameters(self, batch, i_agent):
        
        mask = K.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=K.uint8, device=self.device)

        V = K.zeros((len(batch.state), 1), device=self.device)

        s = K.cat(batch.state, dim=1).to(self.device)
        a = K.cat(batch.action, dim=1).to(self.device)
        r = K.cat(batch.reward, dim=1).to(self.device)
        _a = K.cat(batch.prev_action, dim=1).to(self.device)
        s_ = K.cat([i.to(self.device) for i in batch.next_state if i is not None], dim=1)
        a_ = K.zeros_like(a)[:,0:s_.shape[1],]
        
        W = K.zeros((len(batch.state), 1), device=self.device)
        
        m = K.cat(batch.medium, dim=1).to(self.device)
        t = K.cat(batch.comm_reward, dim=1).to(self.device)
        c = K.cat(batch.comm_action, dim=1).to(self.device)
        m_ = K.zeros_like(m)[:,0:s_.shape[1],]
        c_ = K.zeros_like(c)[:,0:s_.shape[1],]

        if self.comm_actors[0].has_context:
            h = K.cat(batch.comm_context, dim=1).to(self.device)
            h_ = K.cat([i.to(self.device) for i in batch.next_comm_context if i is not None], dim=1)

        if self.normalized_rewards:
            r -= r.mean()
            r /= r.std()
            t -= t.mean()
            t /= t.std()

        R = self.comm_critics[0](K.cat([s[[i_agent],], i_agent*K.ones((1,s.shape[1],1), dtype=self.dtype, device=self.device)], dim=-1),
                                       c[[i_agent],])

        for i in range(self.num_agents):
            if self.comm_actors[0].has_context:
                c_[i,] = self.comm_actors_target[0](K.cat([s_[[i],], i*K.ones((1,s_.shape[1],1), dtype=self.dtype, device=self.device)], dim=-1), h_[[i],])
            else:
                c_[i,] = self.comm_actors_target[0](K.cat([s_[[i],], i*K.ones((1,s_.shape[1],1), dtype=self.dtype, device=self.device)], dim=-1))
        
        W[mask] = self.comm_critics_target[0](K.cat([s_[[i_agent],], i_agent*K.ones((1,s_.shape[1],1), dtype=self.dtype, device=self.device)], dim=-1),
                                                    c_[[i_agent],]).detach()

        #foo1 = t[[i_agent],].squeeze(0)
        #foo2 = r[[i_agent],].squeeze(0)

        #foo1 = (foo1-foo1.min())/(foo1.max()-foo1.min())
        #foo2 = (foo2-foo2.min())/(foo2.max()-foo2.min())
        
        #loss_comm_critic = self.loss_func(R, (W * self.gamma) + foo1 + foo2)
        loss_comm_critic = self.loss_func(R, (W * self.gamma) + t[[i_agent],].squeeze(0) + r[[i_agent],].squeeze(0))
        #loss_comm_critic = self.loss_func(R, (W * self.gamma) + t[[i_agent],].squeeze(0))

        self.comm_critics_optim[0].zero_grad()
        loss_comm_critic.backward()
        K.nn.utils.clip_grad_norm_(self.comm_critics[0].parameters(), 0.5)
        self.comm_critics_optim[0].step()

        for i in range(self.num_agents):
            if self.comm_actors[0].has_context:
                c[i,] = self.comm_actors[0](K.cat([s[[i],], i*K.ones((1,s.shape[1],1), dtype=self.dtype, device=self.device)], dim=-1), h[[i],])
            else:
                c[i,] = self.comm_actors[0](K.cat([s[[i],], i*K.ones((1,s.shape[1],1), dtype=self.dtype, device=self.device)], dim=-1))

        loss_comm_actor = -self.comm_critics[0](K.cat([s[[i_agent],], i_agent*K.ones((1,s.shape[1],1), dtype=self.dtype, device=self.device)], dim=-1), 
                                                      c[[i_agent],]).mean()
        if self.regularization:
            if self.comm_actors[0].has_context:
                loss_comm_actor += (self.comm_actors[0].get_preactivations(K.cat([s[[i_agent],], i_agent*K.ones((1,s.shape[1],1), dtype=self.dtype, device=self.device)], dim=-1),
                                                                           h[[i_agent],])**2).mean()*1e-3
            else:
                loss_comm_actor += (self.comm_actors[0].get_preactivations(K.cat([s[[i_agent],], i_agent*K.ones((1,s.shape[1],1), dtype=self.dtype, device=self.device)], dim=-1))**2).mean()*1e-3

        self.comm_actors_optim[0].zero_grad()        
        loss_comm_actor.backward()
        K.nn.utils.clip_grad_norm_(self.comm_actors[0].parameters(), 0.5)
        self.comm_actors_optim[0].step()

        Q = self.critics[i_agent](K.cat([s[[i_agent],], m], dim=-1),
                                  a[[i_agent],])

        m_ = self.communication.get_m(s_, c_, a[:, mask, :])
        
        for i in range(self.num_agents):
            a_[i,] = gumbel_softmax(self.actors_target[i](K.cat([s_[[i],], m_], dim=-1)), exploration=False)

        V[mask] = self.critics_target[i_agent](K.cat([s_[[i_agent],], m_], dim=-1),
                                               a_[[i_agent],]).detach()

        loss_critic = self.loss_func(Q, (V * self.gamma) + r[[i_agent],].squeeze(0)) 

        self.critics_optim[i_agent].zero_grad()
        loss_critic.backward()
        K.nn.utils.clip_grad_norm_(self.critics[i_agent].parameters(), 0.5)
        self.critics_optim[i_agent].step()

        m = self.communication.get_m(s, c, _a)

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

        soft_update(self.comm_actors_target[0], self.comm_actors[0], self.tau)
        soft_update(self.comm_critics_target[0], self.comm_critics[0], self.tau)
        soft_update(self.actors_target[i_agent], self.actors[i_agent], self.tau)
        soft_update(self.critics_target[i_agent], self.critics[i_agent], self.tau)
        
        return loss_critic.item(), loss_actor.item()
