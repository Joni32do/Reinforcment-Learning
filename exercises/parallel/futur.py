from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def task(name):
    print(f"Task {name} is running")

# Using ThreadPoolExecutor for I/O-bound tasks
with ThreadPoolExecutor(max_workers=5) as executor:
    executor.map(task, [f'Thread-{i}' for i in range(5)])

# Using ProcessPoolExecutor for CPU-bound tasks
with ProcessPoolExecutor(max_workers=5) as executor:
    executor.map(task, [f'Process-{i}' for i in range(5)])
