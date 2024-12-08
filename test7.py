# Caroline Katko, Transy U
# Original code: https://github.com/johnnycode8/dqn_pytorch
# Modified by: Caroline Katko using Chat GPT and Codeium AI
# Source code altered for test 6 (regular DQN) and test 8 (Dueling DQN)
# uses dueling dqn and dynamic epsilion adjustment


#imports
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import random
import torch
from torch import nn
import yaml
import os
import itertools
import torch.nn.functional as F
from collections import deque
import argparse

# Constants
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# classes

# Replay Memory
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

# Dueling DQN
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim1=64, hidden_dim2=96, hidden_dim3=128, hidden_dim4=192, hidden_dim5=128, hidden_dim6=96, hidden_dim7=64):
        super(DuelingDQN, self).__init__()

        # Shared feature extraction layers
        self.fc1 = nn.Linear(state_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, hidden_dim4)
        self.fc5 = nn.Linear(hidden_dim4, hidden_dim5)
        self.fc6 = nn.Linear(hidden_dim5, hidden_dim6)
        self.fc7 = nn.Linear(hidden_dim6, hidden_dim7)

        # Value stream
        self.value_layer = nn.Sequential(
            nn.Linear(hidden_dim7, hidden_dim7),
            nn.ReLU(),
            nn.Linear(hidden_dim7, 1)  # Output: single value V(s)
        )

        # Advantage stream
        self.advantage_layer = nn.Sequential(
            nn.Linear(hidden_dim7, hidden_dim7),
            nn.ReLU(),
            nn.Linear(hidden_dim7, action_dim)  # Output: advantage A(s, a) for each action
        )

    def forward(self, x):
        # Shared feature extraction
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))

        # Compute value and advantage
        value = self.value_layer(x)
        advantage = self.advantage_layer(x)

        # Combine value and advantage into Q-values
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

if __name__ == '__main__':
    state_dim = 3
    action_dim = 2
    net = DuelingDQN(state_dim, action_dim)
    state = torch.randn(10, state_dim)
    output = net(state)
    print(output)


# Agent
class Agent:
    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            hyperparameters = yaml.safe_load(file)[hyperparameter_set]

        self.env_id = hyperparameters['env_id']
        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.discount_factor_g = hyperparameters['discount_factor_g']
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.epsilon_init = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']
        self.stop_on_reward = hyperparameters['stop_on_reward']
        self.enable_double_dqn = hyperparameters['enable_double_dqn']

        self.LOG_FILE = os.path.join(RUNS_DIR, f'{hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{hyperparameter_set}.pt')

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

    def run(self, is_training=True, render=False):
        env = gym.make(self.env_id, render_mode='human' if render else None)
        num_actions = env.action_space.n
        num_states = len(env.observation_space)

        policy_dqn = DuelingDQN(num_states, num_actions).to(device)
        rewards_per_episode = []
        reward_log = []

        if is_training:
            epsilon = self.epsilon_init
            memory = ReplayMemory(self.replay_memory_size)
            target_dqn = DuelingDQN(num_states, num_actions).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
            step_count = 0
            best_reward = float('-inf')
        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()
            total_wins, total_losses, total_draws = 0, 0, 0
            epsilon = 0.0

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)
            terminated = False
            episode_reward = 0.0

            while not terminated:
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(0)).argmax().item()

                new_state, reward, terminated, _, _ = env.step(action)
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)

                episode_reward += reward

                if is_training:
                    memory.append((state, torch.tensor(action), new_state, torch.tensor(reward), terminated))
                    step_count += 1

                state = new_state

            rewards_per_episode.append(episode_reward)
            reward_log.append(episode_reward)

            if not is_training:
                if reward > 0:
                    total_wins += 1
                elif reward < 0:
                    total_losses += 1
                else:
                    total_draws += 1
                print(f"Episode {episode}: Wins: {total_wins}, Losses: {total_losses}, Draws: {total_draws}")
               
                if episode >= 1000:
                    break

            if is_training and episode % 10000 == 0:
                moving_avg_reward = np.mean(reward_log[-100:])
                print(f"Episode {episode}: Moving Average Reward: {moving_avg_reward:.2f}, Epsilon: {epsilon:.4f}")

                if len(reward_log) >= 100:
                    moving_avg_reward = np.mean(reward_log[-100:])
                    if moving_avg_reward > 0:
                        epsilon_decay_factor = 0.999  # Slow decay to allow exploration
                    elif moving_avg_reward < -0.1:
                        epsilon_decay_factor = 0.995  # Faster decay to force exploitation
                    else:
                        epsilon_decay_factor = 0.997  # Moderate decay
                    epsilon = max(epsilon * epsilon_decay_factor, self.epsilon_min)

                if episode_reward > best_reward:
                    best_reward = episode_reward
                    print(f"New best reward {best_reward:.2f} at episode {episode}")
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)

                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                old_epsilon = epsilon  # Save the old epsilon for comparison
                if moving_avg_reward > 0.0:
                    epsilon = max(epsilon * 1.05, self.epsilon_min)  # Increase epsilon
                elif moving_avg_reward < -0.5:
                    epsilon = max(epsilon * 0.95, self.epsilon_min)  # Decrease epsilon
                if epsilon != old_epsilon:
                    print(f"Dynamic epsilon adjustment: Old Epsilon = {old_epsilon:.4f}, New Epsilon = {epsilon:.4f}")
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

                if is_training and moving_avg_reward > 1.0:
                    print(f"Training stopped as moving average reward exceeded 1.0 at episode {episode}.")
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    break

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        states, actions, new_states, rewards, terminations = zip(*mini_batch)
        states = torch.stack(states).float().to(device)
        actions = torch.stack(actions).to(device)
        new_states = torch.stack(new_states).float().to(device)
        rewards = torch.stack(rewards).float().to(device)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            if self.enable_double_dqn:
                best_action = policy_dqn(new_states).argmax(dim=1)
                target_q = rewards + (1 - terminations) * self.discount_factor_g * \
                           target_dqn(new_states).gather(1, best_action.unsqueeze(1)).squeeze()
            else:
                target_q = rewards + (1 - terminations) * self.discount_factor_g * \
                           target_dqn(new_states).max(dim=1)[0]

        current_q = policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze()
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), max_norm=1.0)
        self.optimizer.step()

# main
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test model.')
    parser.add_argument('hyperparameters', help='Hyperparameter set to use')
    parser.add_argument('--train', help='Training mode', action='store_true')
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)

    if args.train:
        dql.run(is_training=True)
    else:
        dql.run(is_training=False, render=True)
