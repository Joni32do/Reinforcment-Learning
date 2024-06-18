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
def discrete(observation):
    dis = [0,0]
    dis[0] = int( round( 20 * (observation[0]+1.2)/1.8))
    dis[1] = int( round( 20 * (observation[1]+0.07)/0.14))
    return dis

def episode(env):
    """ This is an example performing random actions on the environment"""
    while True:
        env.render()
        action = env.action_space.sample()
        print("do action: ", action)
        observation, reward, done, info = env.step(action)
        print("observation: ", observation)
        print("reward: ", reward)
        print("")
        if done:
            break

def q_learning():
    env = gym.make('MountainCar-v0')
    env.reset()
    episode(env)
    env.close()


def main():
    #q_learning()
    env = gym.make('MountainCar-v0')
    env.reset()
    random_episode(env)
    env.close()

if __name__ == "__main__":
    main()
