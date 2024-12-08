# Caroline Katko, Transy U
# Original code: https://github.com/johnnycode8/dqn_pytorch
# Modified by: Caroline Katko using Chat GPT and Codeium AI
# Source code altered for test 2 (regular DQN) and 7 (Dueling DQN)


#imports
import gymnasium as gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import torch
from torch import nn
import yaml
from datetime import datetime, timedelta
import argparse
import itertools
import os
import torch.nn.functional as F
from collections import deque

# Constants
DATE_FORMAT = "%m-%d %H:%M:%S"
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)
matplotlib.use('Agg')

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# classes


# Replay memory
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

# DQN
class DQN(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim1=64, hidden_dim2=96, hidden_dim3=128, hidden_dim4=192, hidden_dim5=128, hidden_dim6=96, hidden_dim7=64):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, hidden_dim3)
        self.fc4 = nn.Linear(hidden_dim3, hidden_dim4)
        self.fc5 = nn.Linear(hidden_dim4, hidden_dim5)
        self.fc6 = nn.Linear(hidden_dim5, hidden_dim6)
        self.fc7 = nn.Linear(hidden_dim6, hidden_dim7)
        self.output = nn.Linear(hidden_dim7, action_dim)
     

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        return self.output(x)


if __name__ == '__main__':
    state_dim = 3
    action_dim = 2
    net = DQN(state_dim, action_dim)
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
        self.fc1_nodes = hyperparameters['fc1_nodes']
        self.fc2_nodes = hyperparameters['fc2_nodes']
        self.fc3_nodes = hyperparameters['fc3_nodes']
        self.fc4_nodes = hyperparameters['fc4_nodes']
        self.fc5_nodes = hyperparameters['fc5_nodes']
        self.fc6_nodes = hyperparameters['fc6_nodes']
        self.fc7_nodes = hyperparameters['fc7_nodes']
        self.enable_double_dqn = hyperparameters['enable_double_dqn']

        self.LOG_FILE = os.path.join(RUNS_DIR, f'{hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{hyperparameter_set}.png')

        self.loss_fn = nn.MSELoss()
        self.optimizer = None

    def run(self, is_training=True, render=False):
        env = gym.make(self.env_id, render_mode='human' if render else None)
        num_actions = env.action_space.n
        num_states = env.observation_space.n if isinstance(env.observation_space, gym.spaces.Discrete) else len(env.observation_space)

        policy_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.fc2_nodes, self.fc3_nodes, self.fc4_nodes, self.fc5_nodes, self.fc6_nodes, self.fc7_nodes).to(device)
        rewards_per_episode = []
        reward_log = []

        if is_training:
            epsilon = self.epsilon_init
            memory = ReplayMemory(self.replay_memory_size)
            target_dqn = DQN(num_states, num_actions, self.fc1_nodes, self.fc2_nodes, self.fc3_nodes, self.fc4_nodes, self.fc5_nodes, self.fc6_nodes, self.fc7_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
            step_count = 0
            best_reward = float('-inf')
        else:
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
            policy_dqn.eval()
            total_wins, total_losses, total_draws = 0, 0, 0    

        for episode in itertools.count():
            state, _ = env.reset()
            state = torch.tensor(state, dtype=torch.float, device=device)
            terminated = False
            episode_reward = 0.0

            while not terminated and episode_reward < self.stop_on_reward:
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = policy_dqn(state.unsqueeze(0)).argmax().item()

                new_state, reward, terminated, _, _ = env.step(action)
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)

                # Reward Shaping
                if not terminated:
                    if action == 0 and new_state[0] >= 18:  # Stand on strong hand
                        reward += 0.2
                    elif action == 1 and new_state[0] > 17:  # Risky hit
                        reward -= 0.1
                    if new_state[0] == 21:  # Bonus for reaching 21
                        reward += 0.2
                else:
                    if new_state[0] > 21:  # Bust
                        reward -= 1.0
                

                episode_reward += reward

                if is_training:
                    memory.append((state, torch.tensor(action), new_state, torch.tensor(reward), terminated))
                    step_count += 1

                state = new_state

            rewards_per_episode.append(episode_reward)
            reward_log.append(episode_reward)

            # Debug: Log episode results
            #print(f"Episode {episode} Reward: {episode_reward:.2f}, Total Rewards: {sum(rewards_per_episode):.2f}")

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
                if episode % 1000 == 0:
                    print("Initialized weights of policy_dqn:")
                    print(next(policy_dqn.parameters()))

            if is_training and episode % 1000000 == 0:
                moving_avg_reward = np.mean(reward_log[-1000:])
                print(f"Episode {episode}: Moving Average Reward: {moving_avg_reward:.2f}, Epsilon: {epsilon:.4f}")

            if is_training and episode % 10000 == 0:

                if episode_reward > best_reward:
                    best_reward = episode_reward
                    print(f"New best reward {best_reward:.2f} at episode {episode}")
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)

                if len(memory) > self.mini_batch_size:
                    mini_batch = memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn)

                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

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

# Main
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


