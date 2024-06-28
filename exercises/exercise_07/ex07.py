import gym
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

def random_episode(env):
    """ This is an example performing random actions on the environment"""
    while True:
        # env.render()
        action = env.action_space.sample()
        print("do action: ", action)
        observation, reward, done, info = env.step(action)
        print("observation: ", observation)
        print("discrete", discrete(observation))
        print("reward: ", reward)
        print("")
        if done:
            break



def normalized_state(s):
    state = np.zeros(2)
    state[0] = (s[0] + 1.2) / 1.8 # position
    state[1] = (s[1] + 0.07) / 0.14 # velocity
    return state


def discrete(s):
    coordinate = np.floor(20 * normalized_state(s)).astype(int)
    return coordinate


def make_feature(s, a):
    coordinate = discrete(s)
    x = np.zeros(1200) # 20 * 20 * 3
    x[coordinate[1] * 20 * 3 + coordinate[0] * 3 + a] = 1
    return x

def make_feature_rbf(s, a, sigma = 0.1):
    norm_state = normalized_state(s)
    x = np.zeros(1200) # 20 * 20 * 3
    for i in range(20):
        for j in range(20):
            for k in range(3):
                val = np.exp(-((i - norm_state[0])**2 + (j - norm_state[1])**2 + (k - a)**2) / (2 * sigma**2))
                if val > 0.01:
                    x[i * 20 * 3 + j * 3 + k] = val


    '''
    x = np.zeros(1200) # 20 * 20 * 3
    i, j, k = np.meshgrid(range(20), range(20), range(3), indexing='ij')
    val = np.exp(-((i - norm_state[0])**2 + (j - norm_state[1])**2 + (k - a)**2) / (2 * sigma**2))
    mask = val > 0.01
    x[mask] = val[mask]
    '''
    return x


def q_value(s, a, w):
    x = make_feature(s, a)
    # x = make_feature_rbf(s, a)
    return x@w

def plot_value_function(w, episode: int):
    # take the maximum of three succeding values
    fig, ax = plt.subplots()
    arr = np.array([max(w[i], w[i+1], w[i+2]) for i in range(0, 1200, 3)])
    arr = np.reshape(arr, (20, 20))
    im = ax.imshow(arr)
    # annotate the value of the value function
    # for i in range(20):
    #     for j in range(20):
    #         ax.text(j, i, round(arr[i, j], 1), ha='center', va='center', color='black')
    fig.colorbar(im, ax=ax, label="Value function")
    ax.set_title(f"Value function for episode {episode}")
    ax.set_xticks([0, 4, 9, 14, 19])
    ax.set_xticklabels([-1.2, -0.6, 0, 0.6, 1.2])
    ax.set_xlabel("Position")
    ax.set_yticks([0, 4, 9, 14, 19])
    ax.set_yticklabels([-0.07, -0.035, 0, 0.035, 0.07])
    ax.set_ylabel("Velocity")
    fig.savefig(f"images/value_function_episode{episode}")
    plt.close(fig)

def plot_repeated_training_investigation(avg_successes_per_episode, avg_steps_per_episode, n_episodes):
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(range(n_episodes), avg_successes_per_episode, label="Average successes per episode")
    axs[1].plot(range(n_episodes), avg_steps_per_episode, color="tab:orange", label="Average steps per episode")
    axs[0].set_xlabel("Episodes")
    axs[1].set_xlabel("Episodes")
    axs[0].set_ylabel("Number of successes")
    axs[1].set_ylabel("Number of steps")
    axs[0].legend()
    axs[1].legend()
    axs[0].grid()
    fig.savefig("images/repeated_training_investigation_long")
    plt.close(fig)



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
    
    done = False
    while not done:
        # env.render()

        if np.random.random() > eps:
            act = best_action(obs, w)
        else:
            act = env.action_space.sample() #epsilon greedy

        obs_next, reward, done, info = env.step(act)
        #print("do action: ", act)
        #print("observation: ", obs)
        #print("reward: ", reward)
        total_reward += reward

        max_action = best_action(obs, w)
        q_current = q_value(obs, act, w)
        q_next = q_value(obs_next, max_action, w)

        error = reward + gamma * q_next - q_current

        # x = np.append(obs, act)
        x = make_feature(obs, act)
        # x = make_feature_rbf(obs, act)
        w = w + alpha * error * x
        obs = obs_next
    return w, total_reward

def q_learning(episodes,eps, alpha, gamma):
    env = gym.make('MountainCar-v0')
    #w = np.zeros([3])
    # w = np.zeros(env.observation_space.shape[0] + 1)
    w = np.zeros((1200)) # 20 states for position, 20 for vel and 3 for actions
    n_successes = 0
    episode_reward = 0
    steps_per_episode = []
    successes_per_episode = []

    for i in tqdm(range(episodes)):
        w, reward = episode(env, w, eps, alpha, gamma)
        # print(f"Episode {i+1} complete")
        # print("Reward=", reward)
        # print("W",w)
        if reward > -200:
            n_successes += 1
        episode_reward += reward

        steps_per_episode.append(-reward)
        successes_per_episode.append(n_successes)

        if i % 200 == 0:
            pass
            # plot_value_function(w, i)
        
    env.close()
    return episode_reward, successes_per_episode, steps_per_episode

def main():
    total_rewards = []
    total_successes = []
    total_steps = []
    n_episodes = 100000

    for i in range(1):
        total, n_successes, steps_per_episode = q_learning(n_episodes, 0.1, 0.05, 0.9)
        total_rewards.append(total)
        total_successes.append(n_successes)
        total_steps.append(steps_per_episode)



        # task b)
        avg_successes_per_episode = np.average(np.array(total_successes), axis=0)
        avg_steps_per_episode = np.average(np.array(total_steps), axis=0)
        np.savez("data/repeated_training_investigation_long", avg_successes_per_episode=avg_successes_per_episode, avg_steps_per_episode=avg_steps_per_episode)
        plot_repeated_training_investigation(avg_successes_per_episode, avg_steps_per_episode, n_episodes)

if __name__ == "__main__":
    main()
