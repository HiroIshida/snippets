import time
import torch
import torch.nn as nn

class FCN(nn.Module):
    def __init__(self):
        super(FCN, self).__init__()
        self.network = nn.Sequential(nn.Linear(500, 100), nn.ReLU(),
            nn.Linear(100, 100), nn.ReLU(),
            nn.Linear(100, 100), nn.ReLU(),
            nn.Linear(100, 50), nn.ReLU(),
            nn.Linear(50, 1)
        )
    def forward(self, x):
        return self.network(x)

class FCNCluster(nn.Module):
    def __init__(self, num_networks):
        super(FCNCluster, self).__init__()
        self.networks = nn.ModuleList([FCN() for _ in range(num_networks)])
        
    def forward(self, x):
        outputs = torch.stack([net(x) for net in self.networks], dim=1)
        max_output, _ = torch.max(outputs, dim=1)
        return max_output

    def jit_compile(self):
        input_tensor = torch.randn(1, 500).cuda()
        traced = torch.jit.trace(self, (input_tensor,))
        model = torch.jit.optimize_for_inference(traced)
        for _ in range(100):  # warm up
            model(input_tensor)
        return model


cluster = FCNCluster(num_networks=50).cuda()
cluster_jit = cluster.jit_compile()
input_tensor = torch.randn(1, 500).cuda()
ts = time.time()
for _ in range(1000):
    cluster_jit(input_tensor)
print(f"time per inference: {(time.time() - ts) / 1000}")
