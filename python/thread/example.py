import threading
import numpy as np
import time


class StateObserver(threading.Thread):
    def __init__(self):
        super(StateObserver, self).__init__()

    def run(self):
        while share_dict["thread_running"]:
            time.sleep(0.4)
            share_dict["obs"] = np.random.randint(10)

class OnlineComputationThread(threading.Thread):
    def __init__(self):
        super(OnlineComputationThread, self).__init__()

    def run(self):
        while share_dict["thread_running"]:
            obs = share_dict["obs"]
            if obs is not None:
                share_dict["res"] = obs + 0.5
                print(share_dict["res"])

share_dict = {"obs": None, "res": None, "thread_running": True}
thread1 = OnlineComputationThread()
thread2 = StateObserver()
thread1.start()
thread2.start()

time.sleep(3.0)
print("killing thread")
share_dict["thread_running"] = False
