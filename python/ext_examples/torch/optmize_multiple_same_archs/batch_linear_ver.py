import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class BatchedLinear(nn.Module):
    def __init__(self, num_networks, in_features, out_features):
        super(BatchedLinear, self).__init__()
        self.num_networks = num_networks
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(num_networks, out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(num_networks, out_features))
        self.reset_parameters()
    
    def reset_parameters(self):
        for i in range(self.num_networks):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
            nn.init.zeros_(self.bias[i])
    
    def forward(self, x):
        x_expanded = x
        weight_t = self.weight.transpose(1, 2)  # (num_networks, in_features, out_features)
        output = torch.einsum('bni,nio->bno', x_expanded, weight_t)
        output += self.bias.unsqueeze(0)
        return output


class BatchedFCN(nn.Module):
    def __init__(self, num_networks):
        super(BatchedFCN, self).__init__()
        self.num_networks = num_networks
        self.linear1 = BatchedLinear(num_networks, 500, 100)
        self.linear2 = BatchedLinear(num_networks, 100, 100)
        self.linear3 = BatchedLinear(num_networks, 100, 100)
        self.linear4 = BatchedLinear(num_networks, 100, 50)
        self.linear5 = BatchedLinear(num_networks, 50, 1)
    
    def forward(self, x):
        # x: (batch_size, 500)
        x = x.unsqueeze(1).expand(-1, self.num_networks, -1)
        x = self.linear1(x)  # (batch_size, num_networks, 100)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.relu(x)
        x = self.linear4(x)
        x = F.relu(x)
        x = self.linear5(x)  # (batch_size, num_networks, 1)
        x = x.squeeze(-1)  # (batch_size, num_networks)
        # Compute max over networks
        max_output, _ = torch.max(x, dim=1)  # (batch_size,)
        return max_output
    
    def jit_compile(self):
        input_tensor = torch.randn(1, 500).cuda()
        traced = torch.jit.trace(self, (input_tensor,))
        model = torch.jit.optimize_for_inference(traced)
        for _ in range(100):  # warm up
            model(input_tensor)
        return model

# Instantiate and compile the batched network
cluster = BatchedFCN(num_networks=50).cuda()
cluster_jit = cluster.jit_compile()
# cluster_jit = cluster
input_tensor = torch.randn(1, 500).cuda()

# Measure inference time
ts = time.time()
for _ in range(1000):
    cluster_jit(input_tensor)
print(f"time per inference: {(time.time() - ts) / 1000}")
