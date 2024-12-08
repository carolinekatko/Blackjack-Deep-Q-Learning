# Caroline Katko, Transy U
# Original code: https://github.com/johnnycode8/dqn_pytorch
# Modified by: Caroline Katko using Chat GPT and Codeium AI
# Source code altered for test 11


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
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# classes

# Prioritized Replay Memory
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

# Dueling DQN
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim1=256, hidden_dim2=256, hidden_dim3=256):
        super(DuelingDQN, self).__init__()
        
        # Shared layers
        self.fc1 = nn.Linear(state_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        
        # Value stream
        self.value_fc = nn.Linear(hidden_dim2, hidden_dim3)
        self.value_output = nn.Linear(hidden_dim3, 1)  # Outputs single value for the state
        
        # Advantage stream
        self.advantage_fc = nn.Linear(hidden_dim2, hidden_dim3)
        self.advantage_output = nn.Linear(hidden_dim3, action_dim)  # Outputs advantage for each action

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Value stream
        value = F.relu(self.value_fc(x))
        value = self.value_output(value)
        
        # Advantage stream
        advantage = F.relu(self.advantage_fc(x))
        advantage = self.advantage_output(advantage)
        
        # Combine streams
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

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
        self.enable_double_dqn = hyperparameters['enable_double_dqn']

        self.LOG_FILE = os.path.join(RUNS_DIR, f'{hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{hyperparameter_set}.png')
        self.memory = PrioritizedReplayMemory(self.replay_memory_size)
        self.loss_fn = nn.MSELoss()
        self.optimizer = None

    def run(self, is_training=True, render=False):
        env = gym.make(self.env_id, render_mode='human' if render else None)
        num_actions = env.action_space.n
        num_states = 3  # Flattened observation space for Blackjack

        # Initialize policy and target networks as Dueling DQN
        policy_dqn = DuelingDQN(num_states, num_actions, self.fc1_nodes, self.fc2_nodes, self.fc3_nodes).to(device)
        rewards_per_episode = []
        reward_log = []
        wins_per_episode = []  # Track wins to calculate win rate
        step_count = 0  # Initialize step count

        if is_training:
            # Training-specific initializations
            target_dqn = DuelingDQN(num_states, num_actions, self.fc1_nodes, self.fc2_nodes, self.fc3_nodes).to(device)
            target_dqn.load_state_dict(policy_dqn.state_dict())
            self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
            self.memory = PrioritizedReplayMemory(self.replay_memory_size)  # Use prioritized replay
            epsilon = self.epsilon_init
            best_reward = float('-inf')
        else:
            # Testing-specific initializations
            policy_dqn.load_state_dict(torch.load(self.MODEL_FILE, map_location=device))
            policy_dqn.eval()
            total_wins, total_losses, total_draws = 0, 0, 0

        # Main training/testing loop
        for episode in itertools.count():
            obs, _ = env.reset()
            state = torch.tensor([obs[0], obs[1], obs[2]], dtype=torch.float, device=device)
            terminated = False
            episode_reward = 0.0
            won = False  # Track if the episode is won
            actions_taken = []

            while not terminated and episode_reward < self.stop_on_reward:
                # Action selection
                if is_training and random.random() < epsilon:
                    action = env.action_space.sample()  # Exploration
                else:
                    with torch.no_grad():
                        q_values = policy_dqn(state.unsqueeze(0))
                        action = q_values.argmax().item()  # Exploitation

                # track action
                actions_taken.append(action)

                # Take action in the environment
                new_obs, reward, terminated, _, _ = env.step(action)
                new_state = torch.tensor([new_obs[0], new_obs[1], new_obs[2]], dtype=torch.float, device=device)
                dealer_card = new_obs[1]

                if not terminated:
                    # Reward for standing on a strong hand
                    if action == 0 and state[0] >= 18:
                        reward += 0.2

                    # Penalty for risky hit
                    if action == 1 and state[0] > 17:
                        reward -= 0.1

                    # Bonus for hitting exactly 21
                    if action == 1 and new_state[0] == 21:
                        reward += 0.2

                    # Reward standing against strong dealer cards
                    if action == 0 and state[0] >= 18 and dealer_card in [10, 11]:
                        reward += 0.1

                if terminated:
                    # Penalty for busting
                    if new_state[0] > 21:
                        reward -= 0.5

                    # Reward Blackjack immediately after deal
                    if new_state[0] == 21 and len(actions_taken) == 1:
                        reward += 1.0

                episode_reward += reward
                if reward > 0:  # Mark win if positive reward occurs
                    won = True

                if is_training:
                    with torch.no_grad():
                        current_q = policy_dqn(state.unsqueeze(0)).squeeze(0)[action]
                        max_target_q = target_dqn(new_state.unsqueeze(0)).max(1)[0]
                        target_q = reward + (1 - terminated) * self.discount_factor_g * max_target_q
                        td_error = target_q - current_q
                        self.memory.append((state, torch.tensor(action), new_state, torch.tensor(reward), terminated), td_error.item())

                    step_count += 1

                state = new_state

            # End of episode processing
            rewards_per_episode.append(episode_reward)
            reward_log.append(episode_reward)
            wins_per_episode.append(1 if won else 0)  # Track win (1) or loss (0)

            # Auto-stop based on win rate
            if is_training and len(wins_per_episode) >= 1000:
                win_rate = np.mean(wins_per_episode[-1000:]) * 100
                if episode % 10000 == 0:
                    print(f"Episode {episode}: Win Rate (last 1000 episodes): {win_rate:.2f}%")
                if win_rate >= 45.0:  # Adjust threshold as needed
                    print(f"Stopping training: Win rate reached {win_rate:.2f}% at episode {episode}.")
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)
                    break

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

            # Periodic logging and model updates during training
            if is_training and episode % 10000 == 0:
                moving_avg_reward = np.mean(reward_log[-1000:])
                print(f"Episode {episode}: Moving Average Reward: {moving_avg_reward:.2f}, Epsilon: {epsilon:.4f}")

                if episode_reward > best_reward:
                    best_reward = episode_reward
                    print(f"New best reward {best_reward:.2f} at episode {episode}. Saving model...")
                    torch.save(policy_dqn.state_dict(), self.MODEL_FILE)

                if len(self.memory) > self.mini_batch_size:
                    mini_batch, indices, weights = self.memory.sample(self.mini_batch_size)
                    self.optimize(mini_batch, policy_dqn, target_dqn, weights, indices)

                epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)

                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

        # Save the final model at the end of training
        if is_training:
            print("Training complete. Saving final model...")
            torch.save(policy_dqn.state_dict(), self.MODEL_FILE)

        env.close()

    def optimize(self, mini_batch, policy_dqn, target_dqn, weights, indices):
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        states = torch.stack(states).float().to(device)
        actions = torch.stack(actions).to(device)
        new_states = torch.stack(new_states).float().to(device)
        rewards = torch.stack(rewards).float().to(device)
        terminations = torch.tensor(terminations).float().to(device)
        weights = weights.clone().detach().to(device)

        with torch.no_grad():
            if self.enable_double_dqn:
                best_actions = policy_dqn(new_states).argmax(dim=1, keepdim=True)
                target_q_values = target_dqn(new_states).gather(1, best_actions).squeeze()
            else:
                target_q_values = target_dqn(new_states).max(dim=1)[0]

            target_q = rewards + (1 - terminations) * self.discount_factor_g * target_q_values

        current_q = policy_dqn(states).gather(1, actions.unsqueeze(1)).squeeze()
        td_errors = target_q - current_q
        loss = (weights * td_errors.pow(2)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update priorities using self.memory
        self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())

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