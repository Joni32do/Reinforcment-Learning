import gym
import numpy as np
import matplotlib.pyplot as plt

def random_episode(env):
    """ This is an example performing random actions on the environment"""
    while True:
        env.render()
        action = env.action_space.sample()
        print("do action: ", action)
        observation, reward, done, info = env.step(action)
        print("observation: ", observation)
        print("discrete", discrete(observation))
        print("reward: ", reward)
        print("")
        if done:
            break

def action(observation, w):
    action_to_take = 1 if np.dot(observation, w) >1 else -1
    return action_to_take

def q(s, a, w):
    x = np.append(s,a)
    q_value = np.dot(w, x)
    return q_value

def best_action(s, w):
    best = -10
    result = -10
    actions = [-1, 0, 1]   #best if taken from environment
    for action in actions:
        best = action if q(s, action, w)>result else best
    return best

def episode(env, w, eps = 0.2, alpha = 0.1, gamma = 0.1):
    obs, reward, done, info = env.step(0)
    obs_old = obs
    q = 0
    while True:
        env.render()

        action = best_action(obs, w) if np.random.random() > eps else env.action_space.sample() #epsilon greedy
        print("do action: ", action)

        obs, reward, done, info = env.step(action)
        print("observation: ", obs)
        print("reward: ", reward)

        max_action = best_action(obs, w)
        q = q(obs_old, action, w) + alpha .* [reward + gamma * q(obs, max_action, w) - q(obs_old, action, w)] #calculate q

        w = w + alpha*(q - q(obs_old, action, w)) * w

        obs_old = obs

        print("")
        if done:
            break
    return w

def q_learning(episodes):
    env = gym.make('MountainCar-v0')
    env.reset()
    w = np.zeros([3])

    for i in range(episodes):
        env.reset()
        w = episode(env,w)
        env.close()


def main():
    q_learning(10)

if __name__ == "__main__":
    main()
