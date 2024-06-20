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


def q_value(s, a, w):
    x = np.append(s, a)
    q_calc = np.dot(w, x)
    return q_calc


def best_action(s, w):
    best = None
    result = -float('inf')
    actions = [0, 1, 2]   #best if taken from environment
    for action in actions:
        q_val = q_value(s, action, w)
        if q_val > result:
            best = action
            result = q_val
    return int(best)


def episode(env, w, eps, alpha, gamma):
    obs = env.reset()
    total_reward = 0
    q = 0
    while True:
        #env.render()

        if np.random.random() > eps:
            act = best_action(obs, w)
        else:
            act = env.action_space.sample() #epsilon greedy

        #print("do action: ", act)

        obs_next, reward, done, info = env.step(act)
        #print("observation: ", obs)
        #print("reward: ", reward)
        total_reward += reward

        max_action = best_action(obs, w)
        q_current = q_value(obs, act, w)
        q_next = q_value(obs_next, max_action ,w)

        error = reward + gamma * q_next - q_current

        x = np.append(obs, act)
        w = w + alpha * error * x

        obs = obs_next

        #print("")
        if done:
            break
    return w, total_reward

def q_learning(episodes,eps, alpha, gamma):
    env = gym.make('MountainCar-v0')
    #w = np.zeros([3])
    w = np.zeros(env.observation_space.shape[0] + 1)
    episode_reward = 0
    for i in range(episodes):
        w, reward = episode(env,w, eps,alpha, gamma)
        #print(f"Episode {i+1} complete")
        #print("Reward=", reward)
        #print("W",w)
        episode_reward += reward
    env.close()
    return episode_reward

def main():
    total = 0
    total = q_learning(20, 0.3, 0.05, 0.9)
    print("total" , total)

if __name__ == "__main__":
    main()
