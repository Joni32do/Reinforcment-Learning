import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

env = gym.make('Blackjack-v0')


def state_to_index(state):
    return state[0] - 12, state[1] - 1, int(state[2])


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
        # print("observation:", obs)
        states.append(obs)
        if obs[0] >= 20:
            # print("stick")
            obs, reward, done, _ = env.step(0)  # step=0 for stick
        else:
            # print("hit")
            obs, reward, done, _ = env.step(1)  # step=1 for hit
        # print("reward:", reward, "\n")
        ret += reward  # Note that gamma = 1. in this exercise
    # print(sur"final observation:", obs)
    return states, ret

def single_run_pi(pi):
    obs = env.reset()  # obs is a tuple: (player_sum, dealer_card, useable_ace)
    done = False
    states = []
    ret = 0.
    while not done:

        if obs[0] <= 11:
            action = 1 # hit
        else:
            states.append(obs)
            action = int(pi[state_to_index(obs)])
        obs, reward, done, _ = env.step(action)
        ret += reward
    return states, ret




def policy_evaluation():
    """ Implementation of first-visit Monte Carlo prediction """
    # suggested dimensionality: player_sum (12-21), dealer card (1-10), useable ace (true/false)
    # possible variables to use:
    V = np.zeros((10, 10, 2))
    returns = np.zeros((10, 10, 2))
    visits = np.zeros((10, 10, 2))
    maxiter = 1000000  # use whatever number of iterations you want
    for i in tqdm(range(maxiter)):
        # always first visit, since there are no cyclic states in a single blackjack game 
        G = 0
        states, ret = single_run_20()
        for state in states:
            V[state_to_index(state)] += ret
            visits[state_to_index(state)] += 1
    V = np.divide(V, visits, where=visits != 0)


    # plot via plt
    fig, ax = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
    player_sum = np.arange(12, 22)
    dealer_card = np.arange(1, 11)
    X, Y = np.meshgrid(player_sum, dealer_card)
    ax[0].plot_surface(X, Y, V[:, :, 0])
    ax[0].set_title("Usable Ace")
    ax[1].plot_surface(X, Y, V[:, :, 1])
    ax[1].set_title("No Usable Ace")
    plt.show()





def monte_carlo_es():
    """ Implementation of Monte Carlo ES """
    # suggested dimensionality: player_sum (12-21), dealer card (1-10), useable ace (true/false)
    # possible variables to use:
    pi = np.zeros((10, 10, 2), dtype=int) # state -> action (hit/stick)
    # Q = np.zeros((10, 10, 2, 2)) # (state (ps 10, dc 10, ua 2), action (hit/stick))
    Q = np.ones((10, 10, 2, 2)) * 100  # recommended: optimistic initialization of Q
    returns = np.zeros((10, 10, 2, 2))
    visits = np.zeros((10, 10, 2, 2))
    maxiter = 10000000  # use whatever number of iterations you want
    for i in tqdm(range(maxiter)):
        # always first visit, since there are no cyclic states in a single blackjack game
        # last reward is the only non-zero reward, therefore we do not have to take track of the rewards

        states, ret = single_run_pi(pi)
        for state in states:
            idx = state_to_index(state)
            idx_action = idx + (pi[idx],)
            returns[idx_action] += ret
            visits[idx_action] += 1
            Q[idx_action] += returns[idx_action] / visits[idx_action]
            # print(Q[idx])
            # print(pi)
            pi[idx] = np.argmax(Q[idx])
            


        if i % 100000 == 0:
            print("Iteration: " + str(i))
            print(pi[:, :, 0])
            print(pi[:, :, 1])
            # print(Q[:, :, 0, 0])
            # print(Q[:, :, 1, 0])
            # print(Q[:, :, 0, 1])
            # print(Q[:, :, 1, 1])

    # fig, ax = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
    # player_sum = np.arange(12, 22)
    # dealer_card = np.arange(1, 11)
    # X, Y = np.meshgrid(player_sum, dealer_card)
    # ax[0].plot_surface(X, Y, V[:, :, 0])
    # ax[0].set_title("Usable Ace")
    # ax[1].plot_surface(X, Y, V[:, :, 1])
    # ax[1].set_title("No Usable Ace")
    # plt.show()


def main():
    # single_run_20()
    # policy_evaluation()
    monte_carlo_es()


if __name__ == "__main__":
    main()
