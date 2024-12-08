# Caroline Katko, Transy U
# Original code: https://gymnasium.farama.org/introduction/train_agent/
# Modified by: Caroline Katko using Chat GPT and Codeium AI
# Preforms regular Q learning


#imports
from collections import defaultdict
import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import argparse


class BlackjackAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []

    def get_action(self, obs: tuple[int, int, bool], is_testing: bool = False) -> int:
        if not is_testing and np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[obs]))

    def update(
        self,
        obs: tuple[int, int, bool],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: tuple[int, int, bool],
    ):
        future_q_value = (not terminated) * np.max(self.q_values[next_obs])
        temporal_difference = (
            reward + self.discount_factor * future_q_value - self.q_values[obs][action]
        )

        self.q_values[obs][action] = (
            self.q_values[obs][action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def train(self, n_episodes: int):
        for episode in tqdm(range(n_episodes)):
            obs, info = self.env.reset()
            done = False

            while not done:
                action = self.get_action(obs)
                next_obs, reward, terminated, truncated, info = self.env.step(action)

                self.update(obs, action, reward, terminated, next_obs)

                done = terminated or truncated
                obs = next_obs

            self.decay_epsilon()

    def test(self, n_episodes: int = 1000):
        print(f"\nTesting the trained agent for {n_episodes} episodes:\n")
        total_rewards = []
        total_wins = 0
        total_losses = 0
        total_draws = 0

        for episode in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            total_reward = 0

            while not done:
                action = self.get_action(obs, is_testing=True)  # Exploit learned policy
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                done = terminated or truncated
                obs = next_obs

            total_rewards.append(total_reward)
            if total_reward > 0:
                total_wins += 1
            elif total_reward < 0:
                total_losses += 1
            else:
                total_draws += 1

        print(f"Testing Completed!")
        print(f"Total Wins: {total_wins}")
        print(f"Total Losses: {total_losses}")
        print(f"Total Draws: {total_draws}")
        print(f"Average Reward: {np.mean(total_rewards):.2f}")



def main(is_testing: bool = False):
    learning_rate = 0.01
    # change n_episodes to how many episodes of training you want
    n_episodes = 10000000
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)
    final_epsilon = 0.1

    env = gym.make("Blackjack-v1", sab=False)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)

    agent = BlackjackAgent(
        env=env,
        learning_rate=learning_rate,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )

    if is_testing:
        agent.test()
    else:
        agent.train(n_episodes)

        # Visualize training results
        fig, axs = plt.subplots(1, 3, figsize=(20, 8))

        axs[0].plot(np.convolve(env.return_queue, np.ones(100), mode="valid"))
        axs[0].set_title("Episode Rewards")
        axs[0].set_xlabel("Episode")
        axs[0].set_ylabel("Reward")

        axs[1].plot(np.convolve(env.length_queue, np.ones(100), mode="valid"))
        axs[1].set_title("Episode Lengths")
        axs[1].set_xlabel("Episode")
        axs[1].set_ylabel("Length")

        axs[2].plot(np.convolve(agent.training_error, np.ones(100), mode="valid"))
        axs[2].set_title("Training Error")
        axs[2].set_xlabel("Episode")
        axs[2].set_ylabel("Temporal Difference")

        plt.tight_layout()
        plt.show()

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test the Blackjack Q-Learning agent.")
    parser.add_argument("--test", help="Run the agent in testing mode", action="store_true")
    args = parser.parse_args()

    main(is_testing=args.test)
