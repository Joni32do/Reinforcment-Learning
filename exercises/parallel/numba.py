from numba import cuda
import numpy as np

@cuda.jit
def gpu_task(arr):
    idx = cuda.grid(1)
    if idx < arr.size:
        arr[idx] += 1

arr = np.zeros(1000000)
threads_per_block = 128
blocks_per_grid = (arr.size + (threads_per_block - 1)) // threads_per_block
gpu_task[blocks_per_grid, threads_per_block](arr)
