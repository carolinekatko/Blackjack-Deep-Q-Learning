# Caroline Katko, Transy U
# Code by Caroline Katko using Chat GPT 
# Dueling DQN

#imports
import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256, 256]):
        super(DuelingDQN, self).__init__()
        
        # Shared layers
        self.shared_layers = nn.ModuleList()
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            self.shared_layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim
        
        # Value stream
        self.value_fc = nn.Linear(hidden_dims[-1], hidden_dims[-1])
        self.value_output = nn.Linear(hidden_dims[-1], 1)
        
        # Advantage stream
        self.advantage_fc = nn.Linear(hidden_dims[-1], hidden_dims[-1])
        self.advantage_output = nn.Linear(hidden_dims[-1], action_dim)

    def forward(self, x):
        # Forward pass through shared layers
        for layer in self.shared_layers:
            x = F.relu(layer(x))
        
        # Value stream
        value = F.relu(self.value_fc(x))
        value = self.value_output(value)
        
        # Advantage stream
        advantage = F.relu(self.advantage_fc(x))
        advantage = self.advantage_output(advantage)
        
        # Combine streams
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values