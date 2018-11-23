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


class PARENT(object):
    def __init__(self, num_agents, observation_space, action_space, medium_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func=F.sigmoid,
                 discrete=True, regularization=False, normalized_rewards=False, communication=None, discrete_comm=True, Comm_Actor=None, Comm_Critic=None, dtype=K.float32, device="cuda"):

        super(PARENT, self).__init__()

        optimizer, lr = optimizer
        actor_lr, critic_lr, comm_actor_lr, comm_critic_lr = lr

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
        self.discrete_comm = discrete_comm
        self.action_space = action_space

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
        if exploration:
            mu += K.tensor(exploration.noise(), dtype=self.dtype, device=self.device)
        
        if self.out_func == F.tanh:
            mu = mu.clamp(-1, 1)
        elif self.out_func == F.sigmoid:
            mu = mu.clamp(0, 1) 
        else:
            mu = mu

        return mu


class MAMDDPG(PARENT):
    def __init__(self, num_agents, observation_space, action_space, medium_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func=F.sigmoid,
                 discrete=True, regularization=False, normalized_rewards=False, communication=None, discrete_comm=True, Comm_Actor=None, Comm_Critic=None, dtype=K.float32, device="cuda"):
    
        super().__init__(num_agents, observation_space, action_space, medium_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func,
                         discrete, regularization, normalized_rewards, communication, discrete_comm, Comm_Actor, Comm_Critic, dtype, device)

        optimizer, lr = optimizer
        actor_lr, critic_lr, comm_actor_lr, comm_critic_lr = lr

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
        self.comm_actors_optim = []

        for i in range(1):
            self.comm_actors.append(Comm_Actor(observation_space*num_agents, medium_space).to(device))
            self.comm_actors_optim.append(optimizer(self.comm_actors[i].parameters(), lr = comm_actor_lr))

        self.entities.extend(self.comm_actors)
        self.entities.extend(self.comm_actors_optim) 


    def select_comm_action(self, state):
        self.comm_actors[0].eval()
        with K.no_grad():
            mu = self.comm_actors[0].get_m(state.to(self.device))
        self.comm_actors[0].train()

        return mu  

    def update_parameters(self, batch, i_agent):
        
        mask = K.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=K.uint8, device=self.device)

        V = K.zeros((len(batch.state), 1), device=self.device)

        s = K.cat(batch.state, dim=1).to(self.device)
        a = K.cat(batch.action, dim=1).to(self.device)
        r = K.cat(batch.reward, dim=1).to(self.device)
        s_ = K.cat([i.to(self.device) for i in batch.next_state if i is not None], dim=1)
        a_ = K.zeros_like(a)[:,0:s_.shape[1],]
        
        m = K.cat(batch.medium, dim=1).to(self.device)
        m_ = K.zeros_like(m)[:,0:s_.shape[1],]

        if self.normalized_rewards:
            r -= r.mean()
            r /= r.std()

        with K.no_grad():
            m = self.comm_actors[0].get_m(s).unsqueeze(0)
            m_ = self.comm_actors[0].get_m(s_).unsqueeze(0)
        
        Q = self.critics[i_agent](K.cat([s[[i_agent],], m], dim=-1),
                                  a[[i_agent],])
        
        for i in range(self.num_agents):
            a_[i,] = self.actors_target[i](K.cat([s_[[i],], m_], dim=-1))

        V[mask] = self.critics_target[i_agent](K.cat([s_[[i_agent],], m_], dim=-1),
                                               a_[[i_agent],]).detach()

        loss_critic = self.loss_func(Q, (V * self.gamma) + r[[i_agent],].squeeze(0)) 

        self.critics_optim[i_agent].zero_grad()
        loss_critic.backward()
        K.nn.utils.clip_grad_norm_(self.critics[i_agent].parameters(), 0.5)
        self.critics_optim[i_agent].step()

        for i in range(self.num_agents):
            a[i,] = self.actors[i](K.cat([s[[i],], m], dim=-1))

        loss_actor = -self.critics[i_agent](K.cat([s[[i_agent],], m], dim=-1), 
                                            a[[i_agent],]).mean()
        
        if self.regularization:
            loss_actor += (self.actors[i_agent].get_preactivations(K.cat([s[[i_agent],], m], dim=-1))**2).mean()*1e-3

        self.actors_optim[i_agent].zero_grad()        
        loss_actor.backward()
        K.nn.utils.clip_grad_norm_(self.actors[i_agent].parameters(), 0.5)
        K.nn.utils.clip_grad_norm_(self.comm_actors[0].parameters(), 0.5)
        self.actors_optim[i_agent].step()

        soft_update(self.actors_target[i_agent], self.actors[i_agent], self.tau)
        soft_update(self.critics_target[i_agent], self.critics[i_agent], self.tau)
        
        loss_medium = self.loss_func(K.cat(list(s), dim=1) , self.comm_actors[0](s))

        self.comm_actors_optim[0].zero_grad()
        loss_medium.backward()
        self.comm_actors_optim[0].step()
        
        return loss_critic.item(), loss_actor.item(), loss_medium.item()


class MAHCDDPG(PARENT):
    def __init__(self, num_agents, observation_space, action_space, medium_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func=F.sigmoid,
                 discrete=True, regularization=False, normalized_rewards=False, communication=None, discrete_comm=True, Comm_Actor=None, Comm_Critic=None, dtype=K.float32, device="cuda"):
        
        super().__init__(num_agents, observation_space, action_space, medium_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func,
                         discrete, regularization, normalized_rewards, communication, discrete_comm, Comm_Actor, Comm_Critic, dtype, device)

        optimizer, lr = optimizer
        actor_lr, critic_lr, comm_actor_lr, comm_critic_lr = lr

        # model initialization
        self.entities = []
        
        # actors
        self.actors = []
        self.actors_target = []
        self.actors_optim = []
        
        for i in range(num_agents):
            self.actors.append(Actor(observation_space+medium_space, action_space, discrete, out_func).to(device))
            self.actors_target.append(Actor(observation_space+medium_space, action_space, discrete, out_func).to(device))
            self.actors_optim.append(optimizer(self.actors[i].parameters(), actor_lr))

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
            self.comm_actors.append(Comm_Actor(observation_space, 1+discrete_comm, discrete_comm, F.sigmoid).to(device))
            self.comm_actors_target.append(Comm_Actor(observation_space, 1+discrete_comm, discrete_comm, F.sigmoid).to(device))
            self.comm_actors_optim.append(optimizer(self.comm_actors[i].parameters(), lr = comm_actor_lr))
            
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
            self.comm_critics.append(Comm_Critic(observation_space*num_agents, (1+discrete_comm)*num_agents).to(device))
            self.comm_critics_target.append(Comm_Critic(observation_space*num_agents, (1+discrete_comm)*num_agents).to(device))
            self.comm_critics_optim.append(optimizer(self.comm_critics[i].parameters(), lr = comm_critic_lr))

        for i in range(num_agents):
            hard_update(self.comm_critics_target[i], self.comm_critics[i]) 

        print('amanin')

    def select_comm_action(self, state, i_agent, exploration=False):
        self.comm_actors[i_agent].eval()
        with K.no_grad():
            mu = self.comm_actors[i_agent](state.to(self.device))
        self.actors[i_agent].train()
        if self.discrete_comm:
            mu = gumbel_softmax(mu, exploration=exploration)
        else:
            if exploration:
                mu += K.tensor(exploration.noise(), dtype=self.dtype, device=self.device)
                mu = mu.clamp(0, 1) 
        return mu

    def update_parameters(self, batch, batch2, i_agent):
        
        mask = K.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=K.uint8, device=self.device)

        V = K.zeros((len(batch.state), 1), device=self.device)

        s = K.cat(batch.state, dim=1).to(self.device)
        a = K.cat(batch.action, dim=1).to(self.device)
        r = K.cat(batch.reward, dim=1).to(self.device)
        s_ = K.cat([i.to(self.device) for i in batch.next_state if i is not None], dim=1)
        a_ = K.zeros_like(a)[:,0:s_.shape[1],]
        
        m = K.cat(batch.medium, dim=1).to(self.device)

        if self.normalized_rewards:
            r -= r.mean()
            r /= r.std()

        Q = self.critics[i_agent](K.cat([s[[i_agent],], m], dim=-1),
                                  a[[i_agent],])
        
        for i in range(self.num_agents):
            a_[i,] = self.actors_target[i](K.cat([s_[[i],], m[:,mask,]], dim=-1))

        V[mask] = self.critics_target[i_agent](K.cat([s_[[i_agent],], m[:,mask,]], dim=-1),
                                               a_[[i_agent],]).detach()

        loss_critic = self.loss_func(Q, (V * self.gamma) + r[[i_agent],].squeeze(0)) 

        self.critics_optim[i_agent].zero_grad()
        loss_critic.backward()
        K.nn.utils.clip_grad_norm_(self.critics[i_agent].parameters(), 0.5)
        self.critics_optim[i_agent].step()

        for i in range(self.num_agents):
            a[i,] = self.actors[i](K.cat([s[[i],], m], dim=-1))

        loss_actor = -self.critics[i_agent](K.cat([s[[i_agent],], m], dim=-1), 
                                            a[[i_agent],]).mean()
        
        if self.regularization:
            loss_actor += (self.actors[i_agent].get_preactivations(K.cat([s[[i_agent],], m], dim=-1))**2).mean()*1e-3

        self.actors_optim[i_agent].zero_grad()        
        loss_actor.backward()
        K.nn.utils.clip_grad_norm_(self.actors[i_agent].parameters(), 0.5)
        self.actors_optim[i_agent].step()

        soft_update(self.actors_target[i_agent], self.actors[i_agent], self.tau)
        soft_update(self.critics_target[i_agent], self.critics[i_agent], self.tau)

        ## update of the communication part

        mask = K.tensor(tuple(map(lambda s: s is not None, batch2.next_state)), dtype=K.uint8, device=self.device)
        V = K.zeros((len(batch2.state), 1), device=self.device)

        s = K.cat(batch2.state, dim=1).to(self.device)
        r = K.cat(batch2.reward, dim=1).to(self.device)
        s_ = K.cat([i.to(self.device) for i in batch2.next_state if i is not None], dim=1)
        
        c = K.cat(batch2.comm_action, dim=1).to(self.device)
        c_ = K.zeros_like(c)[:,0:s_.shape[1],]
        
        if self.normalized_rewards:
            r -= r.mean()
            r /= r.std()

        Q = self.comm_critics[i_agent](s, c)

        for i in range(self.num_agents):
            if self.discrete_comm:
                c_[i,] = gumbel_softmax(self.comm_actors_target[i](s_[[i],]), exploration=False)
            else:
                c_[i,] = self.comm_actors_target[i](s_[[i],])

        V[mask] = self.comm_critics_target[i_agent](s_, c_).detach()

        loss_critic = self.loss_func(Q, (V * self.gamma) + r[[i_agent],].squeeze(0)) 

        self.comm_critics_optim[i_agent].zero_grad()
        loss_critic.backward()
        K.nn.utils.clip_grad_norm_(self.comm_critics[i_agent].parameters(), 0.5)
        self.comm_critics_optim[i_agent].step()

        for i in range(self.num_agents):
            if self.discrete_comm:
                c[i,] = gumbel_softmax(self.comm_actors[i](s[[i],]), exploration=False)
            else:
                c[i,] = self.comm_actors[i](s[[i],])

        loss_actor = -self.comm_critics[i_agent](s, c).mean()
        
        if self.regularization:
            loss_actor += (self.comm_actors[i_agent].get_preactivations(s[[i_agent],])**2).mean()*1e-3

        self.comm_actors_optim[i_agent].zero_grad()        
        loss_actor.backward()
        K.nn.utils.clip_grad_norm_(self.comm_actors[i_agent].parameters(), 0.5)
        self.comm_actors_optim[i_agent].step()

        soft_update(self.comm_actors_target[i_agent], self.comm_actors[i_agent], self.tau)
        soft_update(self.comm_critics_target[i_agent], self.comm_critics[i_agent], self.tau)

        
        return loss_critic.item(), loss_actor.item()


class MAHDDDPG(PARENT):
    def __init__(self, num_agents, observation_space, action_space, medium_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func=F.sigmoid,
                 discrete=True, regularization=False, normalized_rewards=False, communication=None, discrete_comm=True, Comm_Actor=None, Comm_Critic=None, dtype=K.float32, device="cuda"):
        
        super().__init__(num_agents, observation_space, action_space, medium_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func,
                         discrete, regularization, normalized_rewards, communication, discrete_comm, Comm_Actor, Comm_Critic, dtype, device)

        optimizer, lr = optimizer
        actor_lr, critic_lr, comm_actor_lr, comm_critic_lr = lr

        # model initialization
        self.entities = []
        
        # actors
        self.actors = []
        self.actors_target = []
        self.actors_optim = []
        
        for i in range(num_agents):
            self.actors.append(Actor(observation_space+medium_space, action_space, discrete, out_func).to(device))
            self.actors_target.append(Actor(observation_space+medium_space, action_space, discrete, out_func).to(device))
            self.actors_optim.append(optimizer(self.actors[i].parameters(), actor_lr))

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
            self.comm_actors.append(Comm_Actor(observation_space, 1, discrete_comm, F.sigmoid).to(device))
            self.comm_actors_target.append(Comm_Actor(observation_space, 1, discrete_comm, F.sigmoid).to(device))
            self.comm_actors_optim.append(optimizer(self.comm_actors[i].parameters(), lr = comm_actor_lr))
            
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
            self.comm_critics.append(Comm_Critic(observation_space, 1).to(device))
            self.comm_critics_target.append(Comm_Critic(observation_space, 1).to(device))
            self.comm_critics_optim.append(optimizer(self.comm_critics[i].parameters(), lr = comm_critic_lr))

        for i in range(num_agents):
            hard_update(self.comm_critics_target[i], self.comm_critics[i]) 

    def select_comm_action(self, state, i_agent, exploration=False):
        self.comm_actors[i_agent].eval()
        with K.no_grad():
            mu = self.comm_actors[i_agent](state.to(self.device))
        self.comm_actors[i_agent].train()
        if self.discrete_comm:
            mu = gumbel_softmax(mu, exploration=exploration)
        else:
            if exploration:
                mu += K.tensor(exploration.noise(), dtype=self.dtype, device=self.device)
                mu = mu.clamp(0, 1) 
        return mu

    def update_parameters(self, batch, batch2, i_agent):
        
        mask = K.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=K.uint8, device=self.device)

        V = K.zeros((len(batch.state), 1), device=self.device)

        s = K.cat(batch.state, dim=1).to(self.device)
        a = K.cat(batch.action, dim=1).to(self.device)
        r = K.cat(batch.reward, dim=1).to(self.device)
        s_ = K.cat([i.to(self.device) for i in batch.next_state if i is not None], dim=1)
        a_ = K.zeros_like(a)[:,0:s_.shape[1],]
        
        m = K.cat(batch.medium, dim=1).to(self.device)

        if self.normalized_rewards:
            r -= r.mean()
            r /= r.std()

        Q = self.critics[i_agent](K.cat([s[[i_agent],], m], dim=-1),
                                  a[[i_agent],])
        
        a_[i_agent,] = self.actors_target[i_agent](K.cat([s_[[i_agent],], m[:,mask,]], dim=-1))

        V[mask] = self.critics_target[i_agent](K.cat([s_[[i_agent],], m[:,mask,]], dim=-1),
                                               a_[[i_agent],]).detach()

        loss_critic = self.loss_func(Q, (V * self.gamma) + r[[i_agent],].squeeze(0)) 

        self.critics_optim[i_agent].zero_grad()
        loss_critic.backward()
        K.nn.utils.clip_grad_norm_(self.critics[i_agent].parameters(), 0.5)
        self.critics_optim[i_agent].step()

        a[i_agent,] = self.actors[i_agent](K.cat([s[[i_agent],], m], dim=-1))

        loss_actor = -self.critics[i_agent](K.cat([s[[i_agent],], m], dim=-1), 
                                            a[[i_agent],]).mean()
        
        if self.regularization:
            loss_actor += (self.actors[i_agent].get_preactivations(K.cat([s[[i_agent],], m], dim=-1))**2).mean()*1e-3

        self.actors_optim[i_agent].zero_grad()        
        loss_actor.backward()
        K.nn.utils.clip_grad_norm_(self.actors[i_agent].parameters(), 0.5)
        self.actors_optim[i_agent].step()

        soft_update(self.actors_target[i_agent], self.actors[i_agent], self.tau)
        soft_update(self.critics_target[i_agent], self.critics[i_agent], self.tau)

        ## update of the communication part
        mask = K.tensor(tuple(map(lambda s: s is not None, batch2.next_state)), dtype=K.uint8, device=self.device)
        V = K.zeros((len(batch2.state), 1), device=self.device)

        s = K.cat(batch2.state, dim=1).to(self.device)
        r = K.cat(batch2.reward, dim=1).to(self.device)
        s_ = K.cat([i.to(self.device) for i in batch2.next_state if i is not None], dim=1)
        
        c = K.cat(batch2.comm_action, dim=1).to(self.device)
        c_ = K.zeros_like(c)[:,0:s_.shape[1],]
        
        if self.normalized_rewards:
            r -= r.mean()
            r /= r.std()

        Q = self.comm_critics[0](s[[i_agent],], c[[i_agent],])

        if self.discrete_comm:
            c_[i_agent,] = gumbel_softmax(self.comm_actors_target[i_agent](s_[[i_agent],]), exploration=False)
        else:
            c_[i_agent,] = self.comm_actors_target[i_agent](s_[[i_agent],])

        V[mask] = self.comm_critics_target[i_agent](s_[[i_agent],], c_[[i_agent],]).detach()

        loss_critic = self.loss_func(Q, (V * self.gamma) + r[[i_agent],].squeeze(0)) 

        self.comm_critics_optim[i_agent].zero_grad()
        loss_critic.backward()
        K.nn.utils.clip_grad_norm_(self.comm_critics[i_agent].parameters(), 0.5)
        self.comm_critics_optim[i_agent].step()

        if self.discrete_comm:
            c[i_agent,] = gumbel_softmax(self.comm_actors[i_agent](s[[i_agent],]), exploration=False)
        else:
            c[i_agent,] = self.comm_actors[i_agent](s[[i_agent],])

        loss_actor = -self.comm_critics[i_agent](s[[i_agent],], c[[i_agent],]).mean()
        
        if self.regularization:
            loss_actor += (self.comm_actors[i_agent].get_preactivations(s[[i_agent],])**2).mean()*1e-3

        self.comm_actors_optim[i_agent].zero_grad()        
        loss_actor.backward()
        K.nn.utils.clip_grad_norm_(self.comm_actors[i_agent].parameters(), 0.5)
        self.comm_actors_optim[i_agent].step()

        soft_update(self.comm_actors_target[i_agent], self.comm_actors[i_agent], self.tau)
        soft_update(self.comm_critics_target[i_agent], self.comm_critics[i_agent], self.tau)

        
        return loss_critic.item(), loss_actor.item()


class MAHCDDPG_Multi(PARENT):
    def __init__(self, num_agents, observation_space, action_space, medium_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func=F.sigmoid,
                 discrete=True, regularization=False, normalized_rewards=False, communication=None, discrete_comm=True, Comm_Actor=None, Comm_Critic=None, dtype=K.float32, device="cuda"):
        
        super().__init__(num_agents, observation_space, action_space, medium_space, optimizer, Actor, Critic, loss_func, gamma, tau, out_func,
                         discrete, regularization, normalized_rewards, communication, discrete_comm, Comm_Actor, Comm_Critic, dtype, device)

        optimizer, lr = optimizer
        actor_lr, critic_lr, comm_actor_lr, comm_critic_lr = lr

        # model initialization
        self.entities = []
        
        # actors
        self.actors = []
        self.actors_target = []
        self.actors_optim = []
        
        for i in range(num_agents):
            self.actors.append(Actor(observation_space+medium_space, action_space, discrete, out_func).to(device))
            self.actors_target.append(Actor(observation_space+medium_space, action_space, discrete, out_func).to(device))
            self.actors_optim.append(optimizer(self.actors[i].parameters(), actor_lr))

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
            self.comm_actors.append(Comm_Actor(observation_space, num_agents, discrete_comm, F.sigmoid).to(device))
            self.comm_actors_target.append(Comm_Actor(observation_space, num_agents, discrete_comm, F.sigmoid).to(device))
            self.comm_actors_optim.append(optimizer(self.comm_actors[i].parameters(), lr = comm_actor_lr))
            
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
            self.comm_critics.append(Comm_Critic(observation_space*num_agents, num_agents*num_agents).to(device))
            self.comm_critics_target.append(Comm_Critic(observation_space*num_agents, num_agents*num_agents).to(device))
            self.comm_critics_optim.append(optimizer(self.comm_critics[i].parameters(), lr = comm_critic_lr))

        for i in range(num_agents):
            hard_update(self.comm_critics_target[i], self.comm_critics[i]) 

        print('ahanda')

    def select_comm_action(self, state, i_agent, exploration=False):
        self.comm_actors[i_agent].eval()
        with K.no_grad():
            mu = self.comm_actors[i_agent](state.to(self.device))
        self.actors[i_agent].train()
        if self.discrete_comm:
            mu = gumbel_softmax(mu, exploration=exploration)
        else:
            if exploration:
                mu += K.tensor(exploration.noise(), dtype=self.dtype, device=self.device)
                mu = mu.clamp(0, 1) 
        return mu

    def update_parameters(self, batch, batch2, i_agent):
        
        mask = K.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=K.uint8, device=self.device)

        V = K.zeros((len(batch.state), 1), device=self.device)

        s = K.cat(batch.state, dim=1).to(self.device)
        a = K.cat(batch.action, dim=1).to(self.device)
        r = K.cat(batch.reward, dim=1).to(self.device)
        s_ = K.cat([i.to(self.device) for i in batch.next_state if i is not None], dim=1)
        a_ = K.zeros_like(a)[:,0:s_.shape[1],]
        
        m = K.cat(batch.medium, dim=1).to(self.device)

        if self.normalized_rewards:
            r -= r.mean()
            r /= r.std()

        Q = self.critics[i_agent](K.cat([s[[i_agent],], m[[i_agent],]], dim=-1),
                                  a[[i_agent],])
        
        for i in range(self.num_agents):
            a_[i,] = self.actors_target[i](K.cat([s_[[i],], m[[i_agent],][:,mask,]], dim=-1))

        V[mask] = self.critics_target[i_agent](K.cat([s_[[i_agent],], m[[i_agent],][:,mask,]], dim=-1),
                                               a_[[i_agent],]).detach()

        loss_critic = self.loss_func(Q, (V * self.gamma) + r[[i_agent],].squeeze(0)) 

        self.critics_optim[i_agent].zero_grad()
        loss_critic.backward()
        K.nn.utils.clip_grad_norm_(self.critics[i_agent].parameters(), 0.5)
        self.critics_optim[i_agent].step()

        for i in range(self.num_agents):
            a[i,] = self.actors[i](K.cat([s[[i],], m[[i],]], dim=-1))

        loss_actor = -self.critics[i_agent](K.cat([s[[i_agent],], m[[i_agent],]], dim=-1), 
                                            a[[i_agent],]).mean()
        
        if self.regularization:
            loss_actor += (self.actors[i_agent].get_preactivations(K.cat([s[[i_agent],], m[[i_agent],]], dim=-1))**2).mean()*1e-3

        self.actors_optim[i_agent].zero_grad()        
        loss_actor.backward()
        K.nn.utils.clip_grad_norm_(self.actors[i_agent].parameters(), 0.5)
        self.actors_optim[i_agent].step()

        soft_update(self.actors_target[i_agent], self.actors[i_agent], self.tau)
        soft_update(self.critics_target[i_agent], self.critics[i_agent], self.tau)

        ## update of the communication part

        mask = K.tensor(tuple(map(lambda s: s is not None, batch2.next_state)), dtype=K.uint8, device=self.device)
        V = K.zeros((len(batch2.state), 1), device=self.device)

        s = K.cat(batch2.state, dim=1).to(self.device)
        r = K.cat(batch2.reward, dim=1).to(self.device)
        s_ = K.cat([i.to(self.device) for i in batch2.next_state if i is not None], dim=1)
        
        c = K.cat(batch2.comm_action, dim=1).to(self.device)
        c_ = K.zeros_like(c)[:,0:s_.shape[1],]
        
        if self.normalized_rewards:
            r -= r.mean()
            r /= r.std()

        Q = self.comm_critics[i_agent](s, c)

        for i in range(self.num_agents):
            if self.discrete_comm:
                c_[i,] = gumbel_softmax(self.comm_actors_target[i](s_[[i],]), exploration=False)
            else:
                c_[i,] = self.comm_actors_target[i](s_[[i],])

        V[mask] = self.comm_critics_target[i_agent](s_, c_).detach()

        loss_critic = self.loss_func(Q, (V * self.gamma) + r[[i_agent],].squeeze(0)) 

        self.comm_critics_optim[i_agent].zero_grad()
        loss_critic.backward()
        K.nn.utils.clip_grad_norm_(self.comm_critics[i_agent].parameters(), 0.5)
        self.comm_critics_optim[i_agent].step()

        for i in range(self.num_agents):
            if self.discrete_comm:
                c[i,] = gumbel_softmax(self.comm_actors[i](s[[i],]), exploration=False)
            else:
                c[i,] = self.comm_actors[i](s[[i],])

        loss_actor = -self.comm_critics[i_agent](s, c).mean()
        
        if self.regularization:
            loss_actor += (self.comm_actors[i_agent].get_preactivations(s[[i_agent],])**2).mean()*1e-3

        self.comm_actors_optim[i_agent].zero_grad()        
        loss_actor.backward()
        K.nn.utils.clip_grad_norm_(self.comm_actors[i_agent].parameters(), 0.5)
        self.comm_actors_optim[i_agent].step()

        soft_update(self.comm_actors_target[i_agent], self.comm_actors[i_agent], self.tau)
        soft_update(self.comm_critics_target[i_agent], self.comm_critics[i_agent], self.tau)

        
        return loss_critic.item(), loss_actor.item()

