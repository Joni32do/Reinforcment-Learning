import gym
import copy
import random
import numpy as np

class Node:
    def __init__(self, parent=None, action=None):
        self.parent = parent  # parent of this node
        self.action = action  # action leading from parent to this node
        self.children = []
        self.sum_value = 0.  # sum of values observed for this node, use sum_value/visits for the mean
        self.visits = 0


def rollout(env, maxsteps=100):
    """ Random policy for rollouts """
    G = 0
    for i in range(maxsteps):
        action = env.action_space.sample()
        _, reward, terminal, _ = env.step(action)
        G += reward
        if terminal:
            return G
    return G


def mcts(env, root, maxiter=500):
    """ TODO: Use this function as a starting point for implementing Monte Carlo Tree Search
    """
    max_depth = 0
    # this is an example of how to add nodes to the root for all possible actions:
    root.children = [Node(root, a) for a in range(env.action_space.n)]

    for i in range(maxiter):
        state = copy.deepcopy(env)
        G = 0.
        node  = root
        # TODO: traverse the tree using an epsilon greedy tree policy

        """     while node.children:            #going to the bottom of the tree
            if np.random.random() > 0.5:
                node = max( node.children, key = lambda x: x.sum_value)
            else:
                node = random.choice(root.children)
        """
        while node.children:                #going to the bottom of the tree
            values = [c.sum_value/c.visits if c.visits > 0 else 0 for c in node.children] # TODO: Vectorize with np
            bestchild = node.children[np.argmax(values)]
            # epsilon greedy
            if random.random() < 0.5:
                node = random.choice(node.children)
            else:
                node = bestchild

        #Expansion of tree
        if node.visits>1:
            node.children = [Node(node, a) for a in range(env.action_space.n)]

        #Choose random action and take it
            node = random.choice(node.children)
            _, reward, terminal, _ = env.step(node.action)

            G += reward

            if not terminal:
                G += rollout(state)


        # TODO: update all visited nodes in the tree
        depth = 0
        while node is not root:
            node.visits += 1
            node.sum_value += G
            node = node.parent
            depth += 1
        if depth > max_depth: max_depth = depth
    print("max_depth:", max_depth)


def main():
    env = gym.make("Taxi-v3")
    env.seed(0)  # use seed to make results better comparable
    # run the algorithm 10 times:
    rewards = []
    for i in range(10):
        env.reset()
        terminal = False
        root = Node()  # Initialize empty tree
        sum_reward = 0.
        while not terminal:
            env.render()
            mcts(env, root)  # expand tree from root node using mcts
            values = [c.sum_value/c.visits if c.visits > 0 else 0 for c in root.children]  # calculate values for child actions
            bestchild = root.children[np.argmax(values)]  # select the best child
            _, reward, terminal, _ = env.step(bestchild.action) # perform action for child
            root = bestchild  # use the best child as next root
            root.parent = None
            sum_reward += reward
        rewards.append(sum_reward)
        print("finished run " + str(i+1) + " with reward: " + str(sum_reward))
    print("mean reward: ", np.mean(rewards))

if __name__ == "__main__":
    main()
