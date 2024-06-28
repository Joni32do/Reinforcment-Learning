import gym
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from tqdm import tqdm


def print_policy(Q, env):
    """ This is a helper function to print a nice policy from the Q function"""
    moves = [u'←', u'↓', u'→', u'↑']
    if not hasattr(env, 'desc'):
        env = env.env
    dims = env.desc.shape
    policy = np.chararray(dims, unicode=True)
    policy[:] = ' '
    for s in range(len(Q)):
        idx = np.unravel_index(s, dims)
        policy[idx] = moves[np.argmax(Q[s])]
        if env.desc[idx] in [b'H', b'G']:
            policy[idx] = env.desc[idx]
    print('\n'.join([''.join([u'{:2}'.format(item) for item in row]) for row in policy]))


def plot_V(Q, env):
    """ This is a helper function to plot the state values from the Q function"""
    fig = plt.figure()
    if not hasattr(env, 'desc'):
        env = env.env
    dims = env.desc.shape
    V = np.zeros(dims)
    for s in range(len(Q)):
        idx = np.unravel_index(s, dims)
        V[idx] = np.max(Q[s])
        if env.desc[idx] in ['H', 'G']:
            V[idx] = 0.
    plt.imshow(V, origin='upper',
               extent=[0, dims[0], 0, dims[1]], vmin=.0, vmax=.6,
               cmap=plt.cm.RdYlGn, interpolation='none')
    for x, y in product(range(dims[0]), range(dims[1])):
        plt.text(y + 0.5, dims[0] - x - 0.5, '{:.3f}'.format(V[x, y]),
                 horizontalalignment='center',
                 verticalalignment='center')
    plt.xticks([])
    plt.yticks([])


def plot_Q(Q, env):
    """ This is a helper function to plot the Q function """
    from matplotlib import colors, patches
    fig = plt.figure()
    ax = fig.gca()

    if not hasattr(env, 'desc'):
        env = env.env
    dims = env.desc.shape

    up = np.array([[0, 1], [0.5, 0.5], [1, 1]])
    down = np.array([[0, 0], [0.5, 0.5], [1, 0]])
    left = np.array([[0, 0], [0.5, 0.5], [0, 1]])
    right = np.array([[1, 0], [0.5, 0.5], [1, 1]])
    tri = [left, down, right, up]
    pos = [[0.2, 0.5], [0.5, 0.2], [0.8, 0.5], [0.5, 0.8]]

    cmap = plt.cm.RdYlGn
    norm = colors.Normalize(vmin=.0, vmax=.6)

    ax.imshow(np.zeros(dims), origin='upper', extent=[0, dims[0], 0, dims[1]], vmin=.0, vmax=.6, cmap=cmap)
    ax.grid(which='major', color='black', linestyle='-', linewidth=2)

    for s in range(len(Q)):
        idx = np.unravel_index(s, dims)
        x, y = idx
        if env.desc[idx] in ['H', 'G']:
            ax.add_patch(patches.Rectangle((y, 3 - x), 1, 1, color=cmap(.0)))
            plt.text(y + 0.5, dims[0] - x - 0.5, '{:.2f}'.format(.0),
                     horizontalalignment='center',
                     verticalalignment='center')
            continue
        for a in range(len(tri)):
            ax.add_patch(patches.Polygon(tri[a] + np.array([y, 3 - x]), color=cmap(Q[s][a])))
            plt.text(y + pos[a][0], dims[0] - 1 - x + pos[a][1], '{:.2f}'.format(Q[s][a]),
                     horizontalalignment='center', verticalalignment='center',
                     fontsize=9, fontweight=('bold' if Q[s][a] == np.max(Q[s]) else 'normal'))

    plt.xticks([])
    plt.yticks([])

def epsilon_greedy(Q, s, epsilon):
    """ Epsilon-greedy policy """
    if np.random.rand() < epsilon:
        return np.random.randint(4)
    else:
        return np.argmax(Q[s])


def sarsa(env, alpha=0.025, gamma=0.9, epsilon=0.5, num_ep=int(2.5e5), n_step: int = 4):
    Q = np.zeros((env.observation_space.n, env.action_space.n))
    steps_per_episode = np.zeros(num_ep)
    for i in tqdm(range(num_ep)):
        # init episode
        done = False
        t = 0
        tau = 0
        T = np.inf

        s = env.reset()
        states = [s]
        actions = [epsilon_greedy(Q, states[t], epsilon)]
        rewards = [0]  # Placeholder for the first reward

        while tau < T - 1:
            if t < T:
                s_, r, done, _ = env.step(actions[t])
                states.append(s_)
                # actions.append(a)
                rewards.append(r)
                if done:
                    T = t + 1
                else:
                    a_ = epsilon_greedy(Q, s_, epsilon)
                    actions.append(a_)
            tau = t - n_step + 1
            if tau >= 0:
                G = sum([gamma ** (i - tau - 1) * rewards[i] for i in range(tau + 1, min(tau + n_step, T) + 1)])
                # If 
                if tau + n_step < T:
                    G += gamma ** n_step * Q[states[tau + n_step], actions[tau + n_step]]
                Q[states[tau], actions[tau]] += alpha * (G - Q[states[tau], actions[tau]])

            t += 1
            steps_per_episode[i] += 1

    # plot - average over 1% of the episodes
    frac = int(num_ep / 100)
    num_avg = [np.mean(steps_per_episode[i:i + frac]) for i in range(0, num_ep, frac)]
    plt.bar(range(len(num_avg)), num_avg)
    plt.xlabel(f'Average of {frac} Episodes')
    plt.ylabel('Steps')

    return Q


if __name__ == "__main__":

    env = gym.make('FrozenLake-v0', map_name="8x8")
    print("current environment: ")
    env.render()
    print()

    print("Running sarsa...")
    Q = sarsa(env)
    plot_V(Q, env)
    plot_Q(Q, env)
    print_policy(Q, env)
    plt.show()
    # # TODO: run multiple times, evaluate the performance for different n and alpha
    # nstep_sarsa(env)