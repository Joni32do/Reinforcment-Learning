import threading

def task(name):
    print(f"Task {name} is running")

threads = []
for i in range(5):
    t = threading.Thread(target=task, args=(f'Thread-{i}',))
    t.start()
    threads.append(t)

for t in threads:
    t.join()


