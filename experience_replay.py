# Caroline Katko, Transy U
# Original code: https://github.com/johnnycode8/dqn_pytorch
# Experience Replay Memory for DQN


#imports
from collections import deque
import random

class ReplayMemory:
    def __init__(self, maxlen, seed=None):
        self.memory = deque(maxlen=maxlen)

        if seed is not None:
            random.seed(seed)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)
    