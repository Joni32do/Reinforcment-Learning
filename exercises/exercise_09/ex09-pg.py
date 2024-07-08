import gym
import numpy as np
import matplotlib.pyplot as plt


def policy(state, theta):
    """ TODO: return probabilities for actions under softmax action selection """
    action_a = np.exp(np.dot(theta[:,0], state))
    action_b = np.exp(np.dot(theta[:,1], state))

    denom = action_a + action_b
    
    return [action_a/denom, action_b/denom]  # both actions with 0.5 probability => random


def generate_episode(env, theta, display=False):
    """ generates one episode and returns the list of states,
        the list of rewards and the list of actions of that episode """
    state = env.reset()
    states = [state]
    actions = []
    rewards = []
    for t in range(500):
        if display:
            env.render()
        p = policy(state, theta)
        action = np.random.choice(len(p), p=p)

        state, reward, done, info = env.step(action)
        rewards.append(reward)
        actions.append(action)
        if done:
            break
        states.append(state)

    return states, rewards, actions


def REINFORCE(env, use_baseline: bool = False):
    theta = np.random.rand(4, 2)  # policy parameters
    alpha = 0.001
    gamma = 0.99
    episode_lengths = []
    episode_mean_lengths = []

    for e in range(200000):
        if e % 300 == 0:
            states, rewards, actions = generate_episode(env, theta, False)  # display the policy every 300 episodes
        else:
            states, rewards, actions = generate_episode(env, theta, False)
        T = len(states)
        # print("episode: " + str(e) + " length: " + str(len(states)))

        # TODO: keep track of previous 100 episode lengths and compute mean
        if (e+1) % 1000 == 0:
            episode_mean_lengths.append(np.mean(episode_lengths))
            episode_lengths = []
            print(f"{episode_mean_lengths[e//1000]:.3}")
            # print(theta)
        else:
            episode_lengths.append(T)

        

        # TODO: implement the reinforce algorithm to improve the policy weights
        for t in range(T):
            # Could be done 
            G = np.sum([gamma**(k - t - 1)*rewards[k] for k in range(t+1, T)])
            grad = states[t] * (1 - policy(states[t], theta)[actions[t]])
            theta[:,actions[t]] += alpha * gamma**t * G * grad




def main():
    env = gym.make('CartPole-v1')
    REINFORCE(env)
    env.close()


if __name__ == "__main__":
    main()
