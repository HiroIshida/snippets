import time
from r3m import load_r3m
import numpy as np
import torch

r3m = load_r3m("resnet18") # resnet18, resnet34
r3m.eval()

image = np.random.randint(0, 255, (1, 3, 224, 224), dtype=np.uint8)
image = torch.tensor(image).float()

ts = time.time()
output = r3m(image)
print("Time: ", time.time() - ts)

for _ in range(1000):
    ts = time.time()
    output = r3m(image)
    print("Time: ", time.time() - ts)
