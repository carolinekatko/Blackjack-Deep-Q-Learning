# Caroline Katko, Transy U
# Control 1
# Entirely ramdom actions


#imports
import gymnasium as gym
import numpy as np

# Initialize the Blackjack environment
env = gym.make("Blackjack-v1", render_mode=None)

# Track statistics (optional)
total_rewards = []
win_count = 0
loss_count = 0
draw_count = 0

# Play 1000 episodes
num_episodes = 1000
for episode in range(num_episodes):
    observation, info = env.reset()
    done = False
    episode_reward = 0

    while not done:
        # Take a random action
        action = env.action_space.sample()  # Randomly choose Hit (1) or Stick (0)
        observation, reward, done, truncated, info = env.step(action)
        episode_reward += reward

    # Track results
    total_rewards.append(episode_reward)
    if reward == 1:  # Win
        win_count += 1
    elif reward == -1:  # Loss
        loss_count += 1
    else:  # Draw
        draw_count += 1

# Display results
print(f"Total Episodes: {num_episodes}")
print(f"Wins: {win_count}")
print(f"Losses: {loss_count}")
print(f"Draws: {draw_count}")
print(f"Average Reward: {np.mean(total_rewards):.2f}")
