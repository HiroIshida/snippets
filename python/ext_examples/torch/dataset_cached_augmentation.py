import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
np.random.seed(0)
torch.manual_seed(0)

"""
Confirm that updating dataset affects dataloader output
"""

def split_with_ratio(dataset: Dataset, valid_ratio: float = 0.5):
    n_total = len(dataset)  # type: ignore
    n_validate = int(valid_ratio * n_total)
    ds_train, ds_validate = random_split(dataset, [n_total - n_validate, n_validate])
    return ds_train, ds_validate

class IshidaDataset:
    data: np.ndarray
    def __init__(self): self.data  = np.random.randn(10, 2)
    def update(self): self.data  = np.random.randn(10, 2)
    def __len__(self): return len(self.data)
    def __getitem__(self, index): return self.data[index]
    
dataset = IshidaDataset()
dt, dv = split_with_ratio(dataset)

loader = DataLoader(dataset=dt, batch_size=3, shuffle=False)

print("before update")
[print(e) for e in loader]
[print(e) for e in loader]

print("after update")
dataset.update()
[print(e) for e in loader]
[print(e) for e in loader]
