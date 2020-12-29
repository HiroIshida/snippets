import numpy as np
from pyinstrument import Profiler
profiler = Profiler()
profiler.start()
def hoge(i):
    return i + 1
for i in range(1000000):
    hoge(i)
profiler.stop()
print(profiler.output_text(unicode=True, color=True, show_all=True))
