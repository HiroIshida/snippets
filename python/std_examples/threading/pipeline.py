from collections.abc import Callable
import threading
from queue import Queue
from typing import Union, Optional, Callable, List
import numpy as np
import time

class PipelineStage(threading.Thread):
    in_queue: Queue
    out_queue: Queue

    def __init__(self,
                 func,
                 in_stage: Optional["PipelineStage"]):

        super().__init__()
        self.func = func
        if in_stage is None:
            in_queue = Queue()
        else:
            in_queue = in_stage.out_queue
        self.in_queue = in_queue
        self.out_queue = Queue()

    def run(self):
        while True:
            x = self.in_queue.get()
            if x is None:
                self.out_queue.put(None)
                break
            self.out_queue.put(self.func(x))

class Pipeline:
    stages: List[PipelineStage]

    def __init__(self, func_seq: List[Callable]):
        self.stages = []
        for func in func_seq:
            prestage = self.stages[-1] if self.stages else None
            stage = PipelineStage(func, prestage)
            self.stages.append(stage)

    def start(self):
        for stage in self.stages:
            stage.start()

    def join(self):
        for stage in self.stages:
            stage.join()

    def put(self, x):
        self.stages[0].in_queue.put(x)

    def results(self):
        while True:
            val = self.stages[-1].out_queue.get()
            if val is None:
                break
            yield val


if __name__ == "__main__":
    t_sleep = 0.5

    def f1(x: str):
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

    pipeline = Pipeline([f1, f2, f3, f4])
    pipeline.start()
    for _ in range(3):
        pipeline.put("")
    pipeline.put(None)
    for val in pipeline.results():
        print(val)
    pipeline.join()
    print("fin")
