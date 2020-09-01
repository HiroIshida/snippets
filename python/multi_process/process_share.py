# http://kzky.hatenablog.com/entry/2014/12/21/python_multiprocessing

from multiprocessing import Process, Queue
import numpy as np

class myClassA(Process):
    def __init__(self, task_queue, result_queue):
        Process.__init__(self)
        self.daemon = True
        self.a = np.random.randn(100, 100)
        self._task_queue = task_queue
        self._result_queue = result_queue

        self.start()

    def run(self):
        while True:
            next_task = self._task_queue.get()
            if next_task is None:
                print("break the loop")
                break
            ans = next_task["int"] ** 2
            self._result_queue.put(ans)

q_task = Queue()
q_result = Queue()
myClassA(q_task, q_result)
for i in range(100):
    # cannot send lambda as it's not picklable
    dick = {"int": i, "arr": np.random.randn(100)} 
    q_task.put(dick)
    ans = q_result.get()
    print("{0}**2 = {1}".format(i, ans))
