import torch as K
import torch.nn as nn
import torch.nn.functional as F

import pdb


class Critic(nn.Module):
    def __init__(self, observation_space, action_space):
        super(Critic, self).__init__()
        
        input_size = observation_space + action_space
        hidden_size = 128
        output_size = 1

        self.has_context = False
        
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
        self.hidden_size = hidden_size
        self.has_context = True

        self.GRU = nn.GRUCell(input_size, hidden_size)
        self.FC = nn.Linear(hidden_size, output_size)

        
    def forward(self, s, h=None):

        s = K.cat(list(s), dim=1)

        x = s

        if h is None:
            h = self.h
        else:
            h = K.cat(list(h), dim=1)

        h = self.GRU(x, h)

        if self.discrete:
            x = F.softmax(self.FC(h), dim=1)
        else:
            x = self.out_func(self.FC(h))

        self.h = h
        
        return x

    def get_preactivations(self, s, h):

        s = K.cat(list(s), dim=1)
        h = K.cat(list(h), dim=1)
        
        x = s

        h = self.GRU(x, h)
        x = self.FC(h)

        return x

    def init(self, batch_size=1):
        
        h = K.zeros(batch_size, self.hidden_size)

        self.h = h

    def get_h(self):
        
        return self.h

