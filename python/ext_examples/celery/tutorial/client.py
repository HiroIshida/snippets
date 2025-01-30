import numpy as np
import time
from tasks import add, mul, sub, custom_input_output

ts = time.time()
results = [add.delay(4, 6) for _ in range(80)]
print("Waiting for results...")
for result in results:
    print(result.get())
print(f"Time taken: {time.time() - ts}")

inp = {"foo": 1, "bar": (1, 2, 3, 4)}
ts = time.time()
results = [custom_input_output.delay(inp) for _ in range(80)]
print("Waiting for results...")
for result in results:
    print(result.get())
print(f"Time taken: {time.time() - ts}")

