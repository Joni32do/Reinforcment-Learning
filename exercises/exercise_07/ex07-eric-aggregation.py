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


def discret(s):
    coordinate = [0,0]
    coordinate[0] = int(round(19*(s[0]+1.2)/1.8))
    coordinate[1] = int(round(19*(s[1]+0.07)/0.14))
    return coordinate


def weights(s,w):
    coordinate = discret(s)
    factors = w[coordinate[0], coordinate[1], :]
    return factors

def q_value(s, a, w):
    x = np.append(s, a)
    q_calc = np.dot(weights(s,w), x)
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
        q_next = q_value(obs_next, max_action, w)

        error = reward + gamma * q_next - q_current

        x = np.append(obs, act)
        w[discret(obs)] = w[discret(obs)] + alpha * error * x

        obs = obs_next

        #print("")
        if done:
            break
    return w, total_reward

def q_learning(episodes,eps, alpha, gamma):
    env = gym.make('MountainCar-v0')
    aggregate = 20
    #w = np.random.rand(aggregate, aggregate, 3) * 0.01  # Small random values
    w = np.zeros([aggregate,aggregate,3])

    episode_reward = 0
    for i in range(episodes):
        w, reward = episode(env, w, eps, alpha, gamma)
        #print(f"Episode {i+1} complete")
        #print("Reward=", reward)
        #print("W",w)

        episode_reward += reward
    env.close()
    #print("W",w)

    return episode_reward


def main():
    total = 0
    total = q_learning(10000, 0.5, 0.05, 0.95)
    print("total" , total)


if __name__ == "__main__":
    main()

'''

    n_bins = weights.shape[0]
    x = np.linspace(-1.2, 0.6, n_bins)
    y = np.linspace(-0.07, 0.07, n_bins)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(n_bins):
        for j in range(n_bins):
            state = np.array([X[i, j], Y[i, j]])
            Z[i, j] = np.max(weights[discretize(state)])

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Max Q-value')
    ax.set_title('Max Q-values over Discretized State Space')
    plt.show()

   '''
