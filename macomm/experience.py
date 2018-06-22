import random
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def pull(self):
        #memory = self.memory[::-1]
        memory = self.memory
        self.memory = []
        self.position = 0
        return memory

    def __len__(self):
        return len(self.memory)

Transition_Comm = namedtuple('Transition_Comm', ('state', 'action', 'next_state', 'reward', 'medium', 'comm_action', 'comm_reward'))

class ReplayMemoryComm(ReplayMemory):

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition_Comm(*args)
        self.position = (self.position + 1) % self.capacity