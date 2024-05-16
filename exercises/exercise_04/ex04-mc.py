import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

env = gym.make('Blackjack-v0')


def single_run_20():
    """ run the policy that sticks for >= 20 """
    # This example shows how to perform a single run with the policy that hits for player_sum >= 20
    # It can be used for the subtasks
    # Use a comment for the print outputs to increase performance (only there as example)
    obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
    done = False
    states = []
    ret = 0.
    while not done:
        print("observation:", obs)
        states.append(obs)
        if obs[0] >= 20:
            print("stick")
            obs, reward, done, _ = env.step(0)  # step=0 for stick
        else:
            print("hit")
            obs, reward, done, _ = env.step(1)  # step=1 for hit
        print("reward:", reward, "\n")
        ret += reward  # Note that gamma = 1. in this exercise
    print("final observation:", obs)
    return states, ret


def policy_evaluation():
    """ Implementation of first-visit Monte Carlo prediction """
    # suggested dimensionality: player_sum (12-21), dealer card (1-10), useable ace (true/false)
    # possible variables to use:
    V = np.zeros((10, 10, 2))
    returns = np.zeros((10, 10, 2))
    visits = np.zeros((10, 10, 2))
    maxiter = 10000  # use whatever number of iterations you want
    for i in range(maxiter):
        # always first visit, since there are no cyclic states in a single blackjack game 
        G = 0
        states, ret = single_run_20()
        for state in states:
            V[state[0]][state[1]][int(state[2])] += ret
            visits[state[0]][state[1]][int(state[2])] += 1
    V = np.divide(V, visits, where=visits != 0)


    # plot via plt
    fig, ax = plt.subplots(1, 2)
    ax[0].surface(V[:, :, 0])
    ax[0].title("Usable Ace")
    ax[1].surface(V[:, :, 1])
    ax[1].title("No Usable Ace")
    plt.show()





def monte_carlo_es():
    """ Implementation of Monte Carlo ES """
    # suggested dimensionality: player_sum (12-21), dealer card (1-10), useable ace (true/false)
    # possible variables to use:
    pi = np.zeros((10, 10, 2))
    # Q = np.zeros((10, 10, 2, 2))
    Q = np.ones((10, 10, 2, 2)) * 100  # recommended: optimistic initialization of Q
    returns = np.zeros((10, 10, 2, 2))
    visits = np.zeros((10, 10, 2, 2))
    maxiter = 100000000  # use whatever number of iterations you want
    for i in tqdm(range(maxiter)):
        if i % 100000 == 0:
            print("Iteration: " + str(i))
            print(pi[:, :, 0])
            print(pi[:, :, 1])


def main():
    # single_run_20()
    policy_evaluation()
    # monte_carlo_es()


if __name__ == "__main__":
    main()
