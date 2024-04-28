import gymnasium as gym
import numpy as np

custom_map3x3 = [
    'SFF',
    'FFF',
    'FHG',
]
#env = gym.make("FrozenLake-v0", desc=custom_map3x3)

# Init environment
env = gym.make("FrozenLake-v1")

# you can set it to deterministic with:
#env = gym.make("FrozenLake-v0", is_slippery=False)

# If you want to try larger maps you can do this using:
#random_map = gym.envs.toy_text.frozen_lake.generate_random_map(size=5, p=0.8)
#env = gym.make("FrozenLake-v0", desc=random_map)
# Or:
#env = gym.make("FrozenLake-v0", map_name="8x8")


# Init some useful variables:
n_states = env.observation_space.n
n_actions = env.action_space.n


def print_policy(policy, env):
    """ This is a helper function to print a nice policy representation from the policy"""
    moves = [u'←', u'↓', u'→', u'↑']
    if not hasattr(env, 'desc'):
        env = env.env
    dims = env.desc.shape
    pol = np.chararray(dims, unicode=True)
    pol[:] = ' '
    for s in range(len(policy)):
        idx = np.unravel_index(s, dims)
        pol[idx] = moves[policy[s]]
        if env.desc[idx] in [b'H', b'G']:
            pol[idx] = env.desc[idx]
    print('\n'.join([''.join([u'{:2}'.format(item) for item in row]) for row in pol]))


def value_iteration():
    V_states = np.zeros(n_states)  # init values as zero
    theta = 1e-8
    gamma = 0.8
    # TODO: implement the value iteration algorithm
    # Hint: env.P[state][action] gives you tuples (p, n_state, r, is_terminal), which tell you the probability p that you end up in the next state n_state and receive reward r
    delta = theta + 1
    while theta <= delta:
        for s in range(n_states):
            v_old = V_states[s]
            v_tilde = -1
            for a in range(n_actions):
                p, n_state, r, is_terminal = env.P[s][a]
                v_tilde = p*(r + gamma*V_states[s])
                V_states[s] = np.max(V_states[s], v_tilde)

            delta = np.max(delta, abs(V_states[s]-v_old))


    # TODO: After value iteration algorithm, obtain policy and return it
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        for a in range(n_actions):
            # test for all four (or less direction, if less are possible)
            # what the value is and go into the direction of most value
            current_choice_action = 0
            current_choice_value = -np.infty #hack
            p, n_state, r, is_terminal = env.P[s][a]
            if p > 0 and not is_terminal:
                if V_states[n_state] > current_choice_value:
                    current_choice_action = n_state





    return policy


def main():
    # print the environment
    print("current environment: ")
    env.render()
    dims = env.desc.shape
    print()

    # run the value iteration
    policy = value_iteration()
    print("Computed policy: ")
    print(policy.reshape(dims))
    # if you computed a (working) policy, you can print it nicely with the following command:
    print_policy(policy, env)

    # This code can be used to "rollout" a policy in the environment:
    """print ("rollout policy:")
    maxiter = 100
    state = env.reset()
    for i in range(maxiter):
        new_state, reward, done, info = env.step(policy[state])
        env.render()
        state=new_state
        if done:
            print("Finished episode")
            break"""


if __name__ == "__main__":
    main()
