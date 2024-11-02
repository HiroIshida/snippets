import time
import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(500, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )
                
    def forward(self, x):
        return self.network(x)

class FCNCluster(nn.Module):
    def __init__(self, num_networks):
        super(FCNCluster, self).__init__()
        self.networks = nn.ModuleList([FCN() for _ in range(num_networks)])
        
    def forward(self, x):
        outputs = torch.stack([net(x) for net in self.networks], dim=1)  # Shape: (batch_size, num_networks, 1)
        max_output, _ = torch.max(outputs, dim=1)  # Shape: (batch_size, 1)
        return max_output

class FCNNaiveCluster:
    def __init__(self, num_networks):
        self.num_networks = num_networks
        self.networks = [FCN() for _ in range(num_networks)]

    def forward(self, x):
        rets = []
        for net in self.networks:
            ret = net(x).item()
            rets.append(ret)
        return max(rets)

    def __call__(self, x):
        return self.forward(x)

    def cuda(self):
        for net in self.networks:
            net.cuda()

    def jit_compile(self):
        new_networks = []
        dummy_input = torch.randn(1, 500).cuda()
        for net in self.networks:
            net.cuda()
            traced = torch.jit.trace(net, (dummy_input,))
            new_networks.append(torch.jit.optimize_for_inference(traced))
        self.networks = new_networks

        # warm up
        for _ in range(100):
            self(dummy_input)


batch_size = 1
input_tensor = torch.randn(batch_size, 500).cuda()

cluster_naive = FCNNaiveCluster(num_networks=50)
cluster_naive.cuda()
cluster_naive.jit_compile()

cluster = FCNCluster(num_networks=50)
cluster.cuda()

# jit compile
traced = torch.jit.trace(cluster, (input_tensor,))
cluster_jit = torch.jit.optimize_for_inference(traced)
print(f"warm up")
for _ in range(100):
    cluster_jit(input_tensor)
print(f"done")

# bench
ts = time.time()
from pyinstrument import Profiler
profiler = Profiler()
profiler.start()
for _ in range(1000):
    cluster_naive(input_tensor)
profiler.stop()
print(profiler.output_text(unicode=True, color=True, show_all=True))
print(f"time per inference: {(time.time() - ts) / 1000}")

from pyinstrument import Profiler
profiler = Profiler()
profiler.start()
ts = time.time()
for _ in range(1000):
    cluster_jit(input_tensor)
profiler.stop()
print(profiler.output_text(unicode=True, color=True, show_all=True))
print(f"time per inference: {(time.time() - ts) / 1000}")

