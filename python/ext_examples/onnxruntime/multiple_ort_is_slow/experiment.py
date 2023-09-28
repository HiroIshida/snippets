import argparse
import time
import torch
import torch.nn as nn
import onnxruntime as ort

parser = argparse.ArgumentParser()
# parse n_session
parser.add_argument("--session", type=int, default=1)
args = parser.parse_args()

encoder_layers = [
    nn.Conv2d(1, 8, 3, padding=1, stride=(2, 2)),
    nn.ReLU(inplace=True),
    nn.Conv2d(8, 16, 3, padding=1, stride=(2, 2)),
    nn.ReLU(inplace=True),
    nn.Conv2d(16, 32, 3, padding=1, stride=(2, 2)),
    nn.ReLU(inplace=True),
    nn.Conv2d(32, 64, 3, padding=1, stride=(2, 2)),
    nn.ReLU(inplace=True),
    nn.Flatten(),
    nn.Linear(1024, 1000),
    nn.ReLU(inplace=True),
]
encoder = nn.Sequential(*encoder_layers)
filename = "temp.onnx"
dummy_input = torch.zeros(1, 1, 56, 56)
torch.onnx.export(encoder, dummy_input.to(torch.device("cpu")), filename, input_names=["x"], output_names=["y"])

inp = {"x": dummy_input.numpy()}

n_iter, n_session = 100, args.session

# prepare multiple session
sessions = []
for i in range(n_session):
    sessions.append(ort.InferenceSession(filename, providers=['CPUExecutionProvider']))

# pickup one session and run
start = time.time()
session = sessions[0]
for i in range(n_iter * n_session):
    session.run(None, inp)
print("Time per inference (single): ", (time.time() - start) / (n_iter * n_session))

# all session run
start = time.time()
for i in range(n_iter):
    for session in sessions:
        session.run(None, inp)
print("Time per inference (iter): ", (time.time() - start) / n_iter)
