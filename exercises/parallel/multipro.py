from multiprocessing import Process

def task(name):
    print(f"Task {name} is running")

processes = []
for i in range(5):
    p = Process(target=task, args=(f'Process-{i}',))
    p.start()
    processes.append(p)

for p in processes:
    p.join()
