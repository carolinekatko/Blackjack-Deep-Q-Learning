# Caroline Katko, Transy U
# Code by Caroline Katko using Chat GPT and Codeium AI
# prioritized experience replay

#imports
import numpy as np
from collections import deque
import torch

# class
# PrioritizedReplayMemory
class PrioritizedReplayMemory:
    def __init__(self, maxlen, alpha=0.6, epsilon=1e-6):
        self.memory = []
        self.priorities = []
        self.maxlen = maxlen
        self.alpha = alpha
        self.epsilon = epsilon

    def append(self, transition, td_error):
        priority = (abs(td_error) + self.epsilon) ** self.alpha
        self.memory.append(transition)
        self.priorities.append(priority)

        # Ensure both lists do not exceed maxlen
        if len(self.memory) > self.maxlen:
            self.memory.pop(0)
            self.priorities.pop(0)

    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        weights = (len(self.memory) * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        samples = [self.memory[i] for i in indices]
        return samples, indices, torch.tensor(weights, dtype=torch.float)

    def update_priorities(self, indices, td_errors):
        for i, td_error in zip(indices, td_errors):
            if i < len(self.priorities):  # Ensure index validity
                self.priorities[i] = (abs(td_error) + self.epsilon) ** self.alpha

    def __len__(self):
        return len(self.memory)