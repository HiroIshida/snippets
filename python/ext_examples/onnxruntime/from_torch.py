from dataclasses import dataclass
import time
import numpy as np
import torch
import torch.nn as nn
import onnxruntime as ort

layers = nn.Sequential(
        nn.Linear(40, 40),
        nn.ReLU(),
        nn.Linear(40, 40),
        nn.ReLU(),
        nn.Linear(40, 40),
        nn.ReLU(),
        nn.Linear(40, 1),
        nn.Sigmoid())

filename = "temp.onnx"
torch.onnx.export(layers, torch.randn(40, 40).to(torch.device("cpu")), filename, input_names=["x"], output_names=["y"])
ort_sess = ort.InferenceSession(filename, providers=['CPUExecutionProvider'])

ts = time.time()
for _ in range(100):
    out = ort_sess.run(None, {"x": np.random.randn(40, 40).astype(np.float32)})
print((time.time() - ts) / 100)
