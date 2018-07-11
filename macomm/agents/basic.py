import torch as K
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, observation_space, action_space):
        super(Critic, self).__init__()
        
        input_size = observation_space + action_space
        hidden_size = 128
        output_size = 1
        
        self.FC = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, output_size))
        

    def forward(self, s, a):

        s = K.cat(list(s), dim=1)
        a = K.cat(list(a), dim=1)
                
        x = K.cat([s, a], dim=1)        
        x = self.FC(x)
        return x
    
    
class Actor(nn.Module):

    def __init__(self, observation_space, action_space, discrete=True, out_func=F.sigmoid):
        super(Actor, self).__init__()
        
        input_size = observation_space
        hidden_size = 64
        output_size = action_space

        self.discrete = discrete
        self.out_func = out_func
        
        self.FC = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, output_size))
        
    def forward(self, s):

        s = K.cat(list(s), dim=1)

        x = s
        if self.discrete:
            x = F.softmax(self.FC(x), dim=1)
        else:
            x = self.out_func(self.FC(x))
        return x

    def get_preactivations(self, s):

        s = K.cat(list(s), dim=1)
        
        x = s
        x = self.FC(x)
   
        return x
