import time
import torch
import torch.nn as nn
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--cuda", action="store_true", help="cuda")
args = parser.parse_args()
use_cuda = args.cuda
if use_cuda:
    assert torch.cuda.is_available()

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
dummy_input = torch.zeros(1, 1, 56, 56)
if use_cuda:
    encoder = encoder.cuda()
    dummy_input = dummy_input.cuda()

ts = time.time()
for i in range(1000):
    encoder(dummy_input)
print("Time per iter: ", (time.time() - ts) / 1000)

frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(encoder.eval()))
ts = time.time()
for i in range(1000):
    frozen_mod(dummy_input)
print("Time per iter: ", (time.time() - ts) / 1000)
