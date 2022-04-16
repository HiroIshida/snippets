from multiprocessing import Process
import os
import signal
import time
import numpy as np

def dump(obj):
    import pickle
    f = open('hoge.pkl', 'wb')
    try:
        pickle.dump(a, f)
        f.close()
    except:
        print("keyboard intetruppetd but let me dump again")
        f.close()
        with open('hoge.pkl', 'wb') as f:
            pickle.dump(a, f)

a = np.zeros((1000, 300000))
p = Process(target=dump, args=(a,))
p.start()
time.sleep(0.2)
os.kill(p.pid, signal.SIGINT)
print(p.pid)
