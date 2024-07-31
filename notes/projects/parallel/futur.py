from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
import time


def task(name):
    print(f"Task {name} is running")
    x = np.random.rand(100, 100)
    eigs = np.linalg.eigvals(x)
    # time.sleep(3)
    print(f"Task {name} is done")


def main():
    n = 8
    start = time.time()

    # Execute tasks sequentially
    for i in range(n):
        task(f'Task-{i}')
    
    time_seq = time.time()

    # Using ThreadPoolExecutor for I/O-bound tasks
    with ThreadPoolExecutor(max_workers=n) as executor:
        executor.map(task, [f'Thread-{i}' for i in range(n)])

    time_thread = time.time()
    

    # Using ProcessPoolExecutor for CPU-bound tasks
    with ProcessPoolExecutor(max_workers=n) as executor:
        executor.map(task, [f'Process-{i}' for i in range(n)])

    time_process = time.time()

    print(f"Time Sequentially: {time_seq - start:0.2f}")
    print(f"Time Thread: {time_thread - time_seq:0.2f}")
    print(f"Time Process: {time_process - time_thread:0.2f}")


if __name__ == '__main__':
    main()