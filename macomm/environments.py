import torch as K

from maseg.environment import MultiAgentEnv
import maseg.scenarios as scenarios

# def comm_env(observations, comm_actions):
    
#     comm_rewards = K.zeros(len(comm_actions), dtype=observations.dtype).view(-1,1,1)
    
#     if (comm_actions>0.5).sum().item() == 0: # channel stays idle
#         comm_rewards -= 1
#         medium = K.cat([K.zeros_like(observations[[0], ]), K.zeros((1,1,1), dtype=observations.dtype)], dim=-1)
#     elif (comm_actions>0.5).sum().item() > 1: # collision
#         comm_rewards[comm_actions>0.5] -= 1
#         medium = K.cat([K.zeros_like(observations[[0], ]), (len(comm_actions)+1)*K.ones((1,1,1), dtype=observations.dtype)], dim=-1)
#     else:                                     # success
#         granted_agent = K.argmax((comm_actions>0.5)).item()
#         medium = K.cat([observations[[granted_agent], ], (granted_agent+1)*K.ones((1,1,1), dtype=observations.dtype)], dim=-1)

#     comm_rewards = comm_rewards.numpy()
#     medium = medium.numpy()
    
#     return medium, comm_rewards


def comm_env(observations, comm_actions):
    
    comm_rewards = K.zeros((observations.shape[0], observations.shape[1], 1), 
                           dtype=observations.dtype, device=observations.device)
    
    comm_rewards[:,((comm_actions>0.5).sum(dim=0) == 0)[:,0],:] -= 1
    comm_rewards[:,((comm_actions>0.5).sum(dim=0) > 1)[:,0],:] -= 1
    
    medium = K.zeros((1, observations.shape[1], observations.shape[2]+1), 
                     dtype=observations.dtype, device=observations.device)
    

    granted_agent = (comm_actions>0.5).argmax(dim=0)[:,0]
    for i in range(observations.shape[0]):
        medium[:, granted_agent == i, :] = K.cat([observations[[i],][:, granted_agent==i, :], 
                                                  (i+1)*K.ones((1,(granted_agent==i).sum().item(),1), 
                                                               dtype=observations.dtype, device=observations.device)], dim=-1) 

    
    
    if K.is_nonzero(((comm_actions>0.5).sum(dim=0) == 0)[:,0].sum()):
        medium[:,((comm_actions>0.5).sum(dim=0) == 0)[:,0], :] = K.cat([K.zeros((1,1,observations.shape[2]),
                                                                                dtype=observations.dtype, 
                                                                                device=observations.device),
                                                                        K.zeros((1,1,1), 
                                                                                dtype=observations.dtype, 
                                                                                device=observations.device)], 
                                                                       dim=-1)
        
    if K.is_nonzero(((comm_actions>0.5).sum(dim=0) > 1)[:,0].sum()):        
        medium[:,((comm_actions>0.5).sum(dim=0) > 1)[:,0], :] = K.cat([K.zeros((1,1,observations.shape[2]),
                                                                                dtype=observations.dtype, 
                                                                                device=observations.device),
                                                                       K.ones((1,1,1), 
                                                                               dtype=observations.dtype, 
                                                                               device=observations.device)*(len(comm_actions)+1)], 
                                                                      dim=-1)



    return medium, comm_rewards
        

def make_env_cont(scenario_name, benchmark=False):

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:        
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env