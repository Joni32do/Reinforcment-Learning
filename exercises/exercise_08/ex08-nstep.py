import gym
import numpy as np
import matplotlib.pyplot as plt
import threading

def multi_thread_hello_world(n_threads: int=4):
    def hello_world():
        print("Hello World!")
    threads = []
    for i in range(n_threads):
        t = threading.Thread(target=hello_world)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    

def eps_greedy(env, state: int, epsilon: float):
    if np.random.rand() > epsilon:
        a = np.argmax(0)


def nstep_sarsa(env, n=1, alpha=0.1, gamma=0.9, epsilon=0.1, num_ep=int(1e4)):
    """ TODO: implement the n-step sarsa algorithm """
    env.reset()
    # initialize
    
    s = 0

    for i in range(num_ep):
        a = eps_greedy(env, s, epsilon)
        env.step(a)




if __name__ == "__main__":
    multi_thread_hello_world(4)

    # env = gym.make('FrozenLake-v0', map_name="8x8")
    # # TODO: run multiple times, evaluate the performance for different n and alpha
    # nstep_sarsa(env)