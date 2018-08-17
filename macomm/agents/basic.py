import torch as K
import torch.nn as nn
import torch.nn.functional as F


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
        

    def forward(self, s, a, m=None):

        s = K.cat(list(s), dim=1)
        a = K.cat(list(a), dim=1)
        if m is not None:
            m = K.cat(list(m), dim=1)
            x = K.cat([s, a, m], dim=1)
        else:
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
        self.has_context = False
        
        self.FC = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, output_size))
        
    def forward(self, s):

        s = K.cat(list(s), dim=1)

        x = s
        if self.discrete:
            x = F.softmax(self.FC(x), dim=1)
        else:
            if self.out_func == 'linear':
                x = self.FC(x)
            else:
                x = self.out_func(self.FC(x))
        return x

    def get_preactivations(self, s):

        s = K.cat(list(s), dim=1)
        
        x = s
        x = self.FC(x)
   
        return x


class Medium(nn.Module):
    def __init__(self, observation_space, medium_space):
        super(Medium, self).__init__()
        
        input_size = observation_space
        hidden_size = 64
        output_size = medium_space

        self.has_context = False
        
        self.FC_encoder = nn.Sequential(nn.Linear(input_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, output_size))

        self.FC_decoder = nn.Sequential(nn.Linear(output_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, hidden_size), nn.ReLU(True),
                                nn.Linear(hidden_size, observation_space), nn.Tanh()
                                )
        
        

    def forward(self, s):

        s = K.cat(list(s), dim=1)  

        x = self.FC_encoder(s)
        x = self.FC_decoder(x)

        return x

    def get_m(self, s):

        s = K.cat(list(s), dim=1)  

        x = self.FC_encoder(s)

        return x
