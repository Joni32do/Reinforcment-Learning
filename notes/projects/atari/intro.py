# First introduction to Atari games with the OpenAI Gym library

import gymnasium as gym
import numpy as np
from tqdm import tqdm
from time import sleep

# import matplotlib.pyplot as plt


# Init environment
env = gym.make("ALE/Pacman-v5", render_mode="human")

# Init some useful variables:
n_actions = env.action_space.n
n_states = 1  # Atari games have no states, so we set this to 1
r = np.zeros(n_states)  # the r vector is zero everywhere
env.reset()

for i in range(1000):
    env.render()
    env.step(0)
    sleep(0.01)
