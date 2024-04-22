import gymnasium as gym
import numpy as np
from tqdm import tqdm
import subprocess

subprocess.run("export LD=PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so6", shell=True)

# Init environment
# Lets use a smaller 3x3 custom map for faster computations
custom_map3x3 = [
    'SFF',
    'FFF',
    'FHG',
]
env = gym.make("FrozenLake-v1", desc=custom_map3x3, render_mode="human")
# TODO: Uncomment the following line to try the default map (4x4):
# env = gym.make("FrozenLake-v1", render_mode="human")

# Uncomment the following lines for even larger maps:
# random_map = generate_random_map(size=5, p=0.8)
# env = gym.make("FrozenLake-v0", desc=random_map)

# Init some useful variables:
n_states = env.observation_space.n
n_actions = env.action_space.n

r = np.zeros(n_states) # the r vector is zero everywhere except for the goal state (last state)
r[-1] = 1.

gamma = 0.8


""" This is a helper function that returns the transition probability matrix P for a policy """
def trans_matrix_for_policy(policy):
    transitions = np.zeros((n_states, n_states))
    for s in range(n_states):
        probs = env.P[s][policy[s]]
        for el in probs:
            transitions[s, el[1]] += el[0]
    return transitions


""" This is a helper function that returns terminal states """
def terminals():
    terms = []
    for s in range(n_states):
        # terminal is when we end with probability 1 in terminal:
        if env.P[s][0][0][0] == 1.0 and env.P[s][0][0][3] == True:
            terms.append(s)
    return terms


def value_policy(policy):
    P = trans_matrix_for_policy(policy)
    # TODO: calculate and return v
    # (P, r and gamma already given)
    v = np.linalg.solve(np.eye(n_states) - gamma * P, r)
    return v


def bruteforce_policies():
    terms = terminals()
    number_of_non_terminal_states = n_states - len(terms)
    optimalpolicies = []

    # policy = np.zeros(
    #     n_states, dtype=int
    # )  # in the discrete case a policy is just an array with action = policy[state]
    # optimalvalue = np.zeros(n_states)

    # TODO: implement code that tries all possible policies, calculates the values using def value_policy().
    #       Find the optimal values and the optimal policies to answer the exercise questions.
    all_policies = []
    number_of_policies = n_actions**number_of_non_terminal_states

    for i in tqdm(range(number_of_policies)):
        # Initialize an empty list to store the current policy
        policy = []

        # For each state, determine the corresponding action
        # We do this by dividing by the appropriate power of n_actions
        # and taking the modulus with n_actions
        value = i
        for j in range(number_of_non_terminal_states):
            # Find the action for this state
            action = value % n_actions  # Get the remainder to determine the action
            policy.append(action)

            # Update the value to get the next action
            value = value // n_actions

        all_policies.append(policy)

    # fill spots of terminal states with random action (it doesnt matter)
    for term in terms:
        for policy in all_policies:
            policy.insert(term, 1)

    all_policies = np.array(all_policies)

    values = np.array([value_policy(policy) for policy in tqdm(all_policies)])
    values_sum = values.sum(axis=1)
    optimalvalue_index = np.argmax(values_sum)
    optimalvalue = values[optimalvalue_index]

    matches = [np.array_equal(optimalvalue, sub_array) for sub_array in values]
    optimalpolicies = all_policies[matches]
    

    print("Optimal value function:")
    print(optimalvalue)
    print("number optimal policies:")
    print(len(optimalpolicies))
    print("optimal policies:")
    print(np.array(optimalpolicies))

    return optimalpolicies


def main():
    # print the environment
    print("current environment: ")
    env.reset()
    env.render()
    print("")

    # Here a policy is just an array with the action for a state as element
    policy_left = np.zeros(n_states, dtype=int)  # 0 for all states
    policy_right = np.ones(n_states, dtype=int) * 2  # 2 for all states

    # Value functions:
    print("Value function for policy_left (always going left):")
    print(value_policy(policy_left))
    print("Value function for policy_right (always going right):")
    print(value_policy(policy_right))

    optimalpolicies = bruteforce_policies()


    # This code can be used to "rollout" a policy in the environment:
    print("rollout policy:")
    maxiter = 100
    for i in range(10):
        state, _ = env.reset()
        for i in range(maxiter):
            
            new_state, reward, terminated, truncanated, info = env.step(optimalpolicies[0][state])
            env.render()
            state=new_state
            if terminated:
                print("Finished episode")
                break


if __name__ == "__main__":
    main()
