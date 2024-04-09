import numpy as np
import tqdm
import zlib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--compress", action="store_true", help="warm start")
parser.add_argument("--num", type=int, default=1, help="number of workers")

args = parser.parse_args()

model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Conv2d(16, 32, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Flatten(),
    nn.Linear(32 * 14 * 14, 1)
)

class BoxDataset(Dataset):
    def __init__(self, size, compression=True):
        self.size = size
        self.compression = compression
        self.data = []
        self.labels = torch.zeros(size, 1)
        
        for i in tqdm.tqdm(range(size)):
            data = torch.zeros(1, 56, 56)
            x1, y1 = torch.randint(0, 28, (2,))
            x2, y2 = torch.randint(x1.item() + 1, 56, (2,))
            data[0, y1:y2, x1:x2] = 1
            
            if self.compression:
                compressed_data = zlib.compress(data.numpy().tobytes())
                self.data.append(compressed_data)
            else:
                self.data.append(data)
            
            self.labels[i] = (y2 - y1) * (x2 - x1)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        if self.compression:
            compressed_data = self.data[idx]
            data = torch.from_numpy(
                np.frombuffer(zlib.decompress(compressed_data), dtype=np.float32).reshape(1, 56, 56)
            )
        else:
            data = self.data[idx]
        
        return data, self.labels[idx]


torch.manual_seed(42)
dataset = BoxDataset(300000, compression=args.compress)
dataloader = DataLoader(dataset, batch_size=200, shuffle=True, num_workers=args.num)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)
num_epochs = 300

model = model.cuda()

for epoch in range(num_epochs):
    for batch_data, batch_labels in tqdm.tqdm(dataloader):
        batch_data = batch_data.cuda()
        batch_labels = batch_labels.cuda()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

print("Training completed!")
