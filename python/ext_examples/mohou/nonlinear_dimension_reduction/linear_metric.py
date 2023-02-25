from dataclasses import dataclass
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt
from typing import Callable, List, Optional, Tuple
import torch.nn as nn
from torch.nn import Module, Sequential
from mohou.trainer import train, TrainCache, TrainConfig
from mohou.model.common import LossDict, ModelBase, ModelConfigBase


def gram_schmidt(vectors: List[np.ndarray]):
    basis = []
    for vector in vectors:
        for previous_vector in basis:
            vector -= np.dot(vector, previous_vector) * previous_vector
        norm = np.linalg.norm(vector)
        if norm > 0:
            basis.append(vector / norm)
    return np.array(basis)


def create_metric(n_vec: np.ndarray) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    n_dim = len(n_vec)
    vecs = [n_vec]
    for _ in range(n_dim - 1):
        vec = np.random.randn(n_dim)
        vecs.append(vec)
    vec_list = list(gram_schmidt(vecs))

    coefs = [1.0, 2.0] + (n_dim - 2) * [1e-3]

    def dist(x1, x2):
        d = 0.0
        diff = x2 - x1
        for c, v in zip(coefs, vec_list):
            d += c * v.dot(diff) ** 2
        return d

    return dist


@dataclass
class NDRDataset(Dataset):
    dataset: List[Tuple[np.ndarray, np.ndarray, float]]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x1_tmp, x2_tmp, d_tmp = self.dataset[idx]
        x1 = torch.from_numpy(x1_tmp).float()
        x2 = torch.from_numpy(x2_tmp).float()
        d = torch.tensor(d_tmp).float()
        return x1, x2, d


@dataclass
class NDRLinearDataset(NDRDataset):

    @classmethod
    def create(cls) -> "NDRLinearDataset":
        n_dim = 5
        n_vec = np.ones(n_dim) / np.linalg.norm(np.ones(n_dim))
        dist = create_metric(n_vec)
        X = np.random.rand(1000, n_dim)
        centers = np.random.rand(50, n_dim)

        dataset = []
        for center in tqdm.tqdm(centers):
            for x in X:
                d = dist(x, center)
                dataset.append((x, center, d))
        return cls(dataset)


@dataclass
class ProjectionNetConfig(ModelConfigBase):
    n_dim_inp: int 
    n_dim_out: int


class ProjectionNet(ModelBase[ProjectionNetConfig]):
    net: Sequential

    def _setup_from_config(self, config: ProjectionNetConfig) -> None:
        layers = []
        layers.append(nn.Linear(config.n_dim_inp, config.n_dim_inp))
        layers.append(nn.Sigmoid())
        layers.append(nn.Linear(config.n_dim_inp, config.n_dim_inp))
        layers.append(nn.Sigmoid())
        layers.append(nn.Linear(config.n_dim_inp, config.n_dim_out))
        layers.append(nn.Sigmoid())
        self.net = Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)

    def loss(self, dataset: Tuple[torch.Tensor, ...]) -> LossDict:
        X1, X2, dists_target = dataset
        Z1 = self.forward(X1)
        Z2 = self.forward(X2)
        dists = torch.norm(Z1 - Z2, dim=1)
        assert len(dists) == len(dists_target)
        l = nn.MSELoss()(dists, dists_target)
        return LossDict({"loss": l})


if __name__ == "__main__":
    dataset = NDRLinearDataset.create()

    conf = ProjectionNetConfig(5, 2)
    model = ProjectionNet(conf)
    tcache = TrainCache.from_model(model)
    train(Path("/tmp/hoge"), tcache, dataset, device=torch.device("cpu"))
