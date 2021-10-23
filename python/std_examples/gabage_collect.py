import numpy as np
import copy
import os
import gc
import psutil

proc = psutil.Process(os.getpid())
gc.collect()

mem0 = proc.memory_info().rss

def show_memory(msg=""):
    mem_now = proc.memory_info().rss
    print(msg + "memory: {}".format(mem_now - mem0))

class Holder:
    def __init__(self, data=None):
        if data is None:
            self.data = np.random.randn(100, 10000)
        else:
            self.data = data

    def split1(self):
        print("inside split1")
        h1 = copy.deepcopy(self)
        h1.data = copy.deepcopy(self.data[:50, :])
        show_memory()

        h2 = copy.deepcopy(self)
        h2.data = copy.deepcopy(self.data[50:, :])
        show_memory()
        return h1, h2

    def split2(self):
        print("inside split2")
        h1 = Holder(copy.deepcopy(self.data[:50, :]))
        show_memory()
        h2 = Holder(copy.deepcopy(self.data[50:, :]))
        show_memory()
        return h1, h2

show_memory()

h = Holder()
show_memory()

h1, h2 = h.split1()
del h
gc.collect()
show_memory()
