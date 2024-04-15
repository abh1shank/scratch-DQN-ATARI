import gymnasium as gym
import numpy as np
from collections import deque
from DQN import DeepQLearning
import matplotlib.pyplot as plt

GAMMA = 0.99
EPSILON = 1.0
NUM_EPS = 750

env = gym.make("ALE/MsPacman-ram-v5")
observation_space = env.action_space
print(observation_space)
agent = DeepQLearning(env, GAMMA, EPSILON, NUM_EPS)
agent.training_episodes()
agent.main_net.save('pacman_dqn_model.h5')
plt.plot(range(len(agent.sum_rewards)), agent.sum_rewards)
plt.title('Episodes vs Sum of Rewards')
plt.xlabel('Episode')
plt.ylabel('Sum of Rewards')
plt.grid(True)
plt.savefig('episodes_vs_rewards_plot.png')
plt.show()