import time
import tqdm
import torch
import torch.nn as nn


class EncoderTorch(nn.Module):
    def __init__(self, n_channel: int, dim_bottleneck: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(n_channel, 8, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1, stride=2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 16, dim_bottleneck),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.model(x)


def measure(optimized_model):
    for _ in range(10):
        torch_output = optimized_model(batch)
    ts = time.time()
    for _ in tqdm.tqdm(range(30000)):
        torch_output = optimized_model(batch)
    print(f"Time taken: {time.time() - ts:.2f} seconds")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float32
    torch_model = EncoderTorch(n_channel=1, dim_bottleneck=200).to(device, dtype)
    batch = torch.randn(1, 1, 122, 122, device=device, dtype=dtype)

    print(f"measure original model")
    measure(torch_model)

    print("measure jit.trace model")
    traced = torch.jit.trace(torch_model.eval(), (batch,))
    optimized_model = torch.jit.optimize_for_inference(traced)
    measure(optimized_model)

    print("measure compiled model")
    optimized_model = torch.compile(torch_model, fullgraph=True, mode="max-autotune")
    measure(optimized_model)
