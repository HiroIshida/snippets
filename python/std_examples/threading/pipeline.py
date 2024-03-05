import threading
from queue import Queue
import numpy as np
import time

t_sleep = 0.2

def f1():
    time.sleep(t_sleep)
    return "my "

def f2(x: str):
    time.sleep(t_sleep)
    return x + "name "

def f3(x: str):
    time.sleep(t_sleep)
    return x + "is "

def f4(x: str):
    time.sleep(t_sleep)
    return x + "Ishida"

f0_result_queue = Queue()
f1_result_queue = Queue()
f2_result_queue = Queue()
f3_result_queue = Queue()
f4_result_queue = Queue()

def f1_thread():
    while True:
        x = f0_result_queue.get()
        if x is None:
            f1_result_queue.put(None)
            break
        f1_result_queue.put(f1())

def f2_thread():
    while True:
        x = f1_result_queue.get()
        if x is None:
            f2_result_queue.put(None)
            break
        f2_result_queue.put(f2(x))


def f3_thread():
    while True:
        x = f2_result_queue.get()
        if x is None:
            f3_result_queue.put(None)
            break
        f3_result_queue.put(f3(x))


def f4_thread():
    while True:
        x = f3_result_queue.get()
        if x is None:
            f4_result_queue.put(None)
            break
        f4_result_queue.put(f4(x))


# naively
ts = time.time()
for _ in range(5):
    f4(f3(f2(f1())))
print(f"Time: {time.time() - ts}")


# pipeline
t1 = threading.Thread(target=f1_thread)
t2 = threading.Thread(target=f2_thread)
t3 = threading.Thread(target=f3_thread)
t4 = threading.Thread(target=f4_thread)
t1.start()
t2.start()
t3.start()
t4.start()

ts = time.time()
for _ in range(5):
    f0_result_queue.put(1)
f0_result_queue.put(None)

t1.join()
t2.join()
t3.join()
t4.join()
while not f4_result_queue.empty():
    print(f"Result: {f4_result_queue.get()}")
print(f"Time: {time.time() - ts}")
