import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from mohou.model.common import LossDict, ModelBase, ModelConfigBase
from mohou.trainer import TrainCache, TrainConfig, train

from typing import Any, Optional, TypeVar, Generic, List, Tuple, Literal


@dataclass
class ModelConfig(ModelConfigBase):
    n_middle: int = 200
    n_layers: int = 3
    monotonic: Optional[Literal["square", "exp", "abs"]] = None


class MaybeMonotonicModel(ModelBase[ModelConfig]):
    net: nn.Sequential

    class PositiveLinearBySquare(nn.Linear):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return F.linear(input, self.weight ** 2, self.bias)

    class PositiveLinearByExp(nn.Linear):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return F.linear(input, self.weight.exp(), self.bias)

    class PositiveLinearByAbs(nn.Linear):
        def forward(self, input: torch.Tensor) -> torch.Tensor:
            return F.linear(input, torch.abs(self.weight), self.bias)

    def _setup_from_config(self, config: ModelConfig) -> None:

        if config.monotonic == "square":
            linear_type = self.PositiveLinearBySquare
        elif config.monotonic == "exp":
            linear_type = self.PositiveLinearByExp
        elif config.monotonic == "abs":
            linear_type = self.PositiveLinearByAbs
        elif config.monotonic is None:
            linear_type = nn.Linear
        else:
            assert False, "no such {}".format(config.monotonic)

        layers = [
            linear_type(1, config.n_middle),
            nn.Sigmoid()
        ]
        for i in range(config.n_layers - 2):
            layers.append(linear_type(config.n_middle, config.n_middle))
            layers.append(nn.Sigmoid())
        layers.append(linear_type(config.n_middle, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.net(X)

    def loss(self, dataset: Tuple[torch.Tensor, torch.Tensor]) -> LossDict:
        X, Y = dataset
        Y_hat = self.forward(X)
        loss = nn.MSELoss()(Y_hat, Y)
        return LossDict({"loss": loss})


@dataclass
class MaybeMonotonicDataset(Dataset):
    dataset: List[Tuple[float, float]]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        x_, y_ = self.dataset[idx]
        x = torch.Tensor([x_]).float()
        y = torch.Tensor([y_]).float()
        return x, y


if __name__ == "__main__":
    data_list = []
    for i in range(500):
        x = np.random.randn() * 2
        y = np.sin(x * 2.0) + np.random.randn() * 0.3 + x
        data_list.append((x, y))
    dataset = MaybeMonotonicDataset(data_list)

    models = dict()
    for monotonic_name in ["none", "square", "abs", "exp"]:
        monotonic = None if monotonic_name == "none" else monotonic_name
        model = MaybeMonotonicModel(ModelConfig(monotonic=monotonic), torch.device("cpu"))
        tcache = TrainCache.from_model(model)
        tconf = TrainConfig(n_epoch=400, learning_rate=0.01)

        with TemporaryDirectory() as td:
            td_path = Path(td)
            train(td_path, tcache, dataset=dataset, config=tconf, device=torch.device("cpu"))
        print(tcache.min_valid_loss)
        models[monotonic_name] = model

    # plot
    arr = np.array(data_list)
    plt.scatter(arr[:, 0], arr[:, 1], c="black", s=1.0)

    xlin = np.linspace(-10, 10, 1000)
    for name, model in models.items():
        X = torch.from_numpy(xlin).float().unsqueeze(dim=1)
        ylin = model.forward(X).squeeze().cpu().detach().numpy()
        plt.plot(xlin, ylin, label=name, lw=2.0)
    plt.xlim(-7, 7)
    plt.ylim(-6, 6)
    plt.legend()
    plt.show()
