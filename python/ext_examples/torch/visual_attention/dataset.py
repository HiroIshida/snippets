import matplotlib.pyplot as plt
import argparse
import tqdm
from rpbench.two_dimensional.bubbly_world import BubblyWorldSimple
import numpy as np
import torch
from typing import List
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class Circle:
    x: np.ndarray
    r: float

    def sdf(self, x: np.ndarray) -> float:
        return np.linalg.norm(x - self.x) - self.r


@dataclass
class BubblyWorld:
    circles: List[Circle]

    @classmethod
    def sample(cls) -> "BubbleWorld":
        n_circle = 2
        circles = []
        for _ in range(n_circle):
            x = np.random.rand(2)
            r = np.random.rand() * 0.1
            circles.append(Circle(x, r))
        return cls(circles)

    def sdf(self, x: np.ndarray) -> float:
        return np.min([c.sdf(x) for c in self.circles])

    def get_grid_map(self, xlin: np.ndarray, ylin: np.ndarray) -> np.ndarray:
        X, Y = np.meshgrid(xlin, ylin)
        X = X.reshape(-1)
        Y = Y.reshape(-1)
        X = np.stack([X, Y], axis=1)
        Z = np.array([self.sdf(x) for x in X])
        Z = Z.reshape(len(xlin), len(ylin))
        return Z


class BubblyWorldDataset(torch.utils.data.Dataset):
    gridmap_list: List[torch.Tensor]
    X: List[torch.Tensor]
    y: List[torch.Tensor]
    n_sample_per_world: int
    def __init__(self, n_sample: int, n_sampler_per_world: int = 10):
        self.n_sample_per_world = n_sampler_per_world
        self.gridmap_list = []
        self.X = []
        self.y = []

        for i in tqdm.tqdm(range(n_sample)):
            world = BubblyWorldSimple.sample()
            binary_gridmap: np.ndarray = world.get_grid_map() > 0
            tmp = torch.from_numpy(binary_gridmap).float()
            self.gridmap_list.append(tmp.unsqueeze(0))

            sdf = world.get_exact_sdf()
            for j in range(n_sampler_per_world):
                x = np.random.rand(2)
                self.X.append(torch.from_numpy(x).float())

                distance = sdf(np.expand_dims(x, axis=0))[0]
                self.y.append(torch.Tensor([distance]).float())

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        idx_gridmap = idx // self.n_sample_per_world
        return self.gridmap_list[idx_gridmap], self.X[idx], self.y[idx]


class BubblyWorldNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        bottle_neck = 300
        x_dim = 2
        n_channel = 1
        self.encoder_layers = []

        encoder_layers = [
            nn.Conv2d(n_channel, 8, 3, padding=1, stride=(2, 2)),  # 14x14
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 16, 3, padding=1, stride=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, padding=1, stride=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, stride=(2, 2)),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(1024, bottle_neck),
            nn.ReLU(inplace=True),
        ]
        self.encoder = nn.Sequential(*encoder_layers)

        # layer that magnitude of x
        magni_layer = [
            nn.Linear(x_dim, bottle_neck),
            nn.ReLU(inplace=True),
            nn.Linear(bottle_neck, bottle_neck),
            nn.ReLU(inplace=True),
        ]
        fc_layer = [
            nn.Linear(bottle_neck + bottle_neck, bottle_neck),
            nn.ReLU(inplace=True),
            nn.Linear(bottle_neck, bottle_neck),
            nn.ReLU(inplace=True),
            nn.Linear(bottle_neck, 1),
        ]
        self.fc_layer = nn.Sequential(*fc_layer)
        self.magni_layer = nn.Sequential(*magni_layer)

    def forward(self, gridmaps: torch.Tensor, xs: torch.Tensor):
        gridmaps = self.encoder(gridmaps)
        xs = self.magni_layer(xs)
        x = torch.cat([gridmaps, xs], dim=1)
        return self.fc_layer(x)

    def loss(self, gridmaps: torch.Tensor, xs: torch.Tensor, ys: torch.Tensor):
        y_preds = self.forward(gridmaps, xs)
        return torch.nn.functional.mse_loss(y_preds, ys)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
        
    if args.train:
        dataset = BubblyWorldDataset(1000, 300)

        # train test split
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
        # dataloader
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=200, shuffle=True, num_workers=4)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=200, shuffle=True, num_workers=4)
        model = BubblyWorldNet()

        # train
        train_loss_history = []
        test_loss_history = []

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        for epoch in range(20):
            model.train()
            loss_list = []
            for sample in tqdm.tqdm(train_dataloader):
                optimizer.zero_grad()
                loss = model.loss(*sample)
                loss.backward()
                optimizer.step()
                loss_list.append(loss.detach().cpu().numpy().item())
            print(f"epoch: {epoch}, train loss: {np.mean(loss_list)}")
            train_loss_history.append(np.mean(loss_list))

            # test
            model.eval()
            loss_list = []
            for sample in tqdm.tqdm(test_dataloader):
                loss = model.loss(*sample)
                loss_list.append(loss.detach().cpu().numpy().item())
            print(f"epoch: {epoch}, test loss: {np.mean(loss_list)}")
            test_loss_history.append(np.mean(loss_list))

            # save if the test loss is the best
            if epoch == 0 or np.mean(loss_list) < np.min(test_loss_history):
                torch.save(model.state_dict(), f"model_best.pth")

        plt.plot(train_loss_history, label="train")
        plt.plot(test_loss_history, label="test")
        plt.legend()
        plt.show()
    else:
        world = BubblyWorldSimple.sample()
        gmap = world.get_grid_map()
        xlin, ylin = np.linspace(0, 1, 56), np.linspace(0, 1, 56)
        X, Y = np.meshgrid(xlin, ylin)
        X = X.reshape(-1)
        Y = Y.reshape(-1)
        X = np.stack([X, Y], axis=1)
        X_torch = torch.from_numpy(X).float()
        # load 
        model = BubblyWorldNet()
        model.load_state_dict(torch.load("model_best.pth"))
        model.eval()
        torch_map = torch.from_numpy(gmap).unsqueeze(0).unsqueeze(0).float()
        torch_map = torch_map.repeat(X_torch.shape[0], 1, 1, 1)
        Z = model.forward(torch_map, X_torch).detach().numpy()
        Z = Z.reshape(56, 56)
        # fig, axes = plt.subplots((1, 2))
        fig, axes = plt.subplots(1, 2)
        # show predicted
        X, Y = np.meshgrid(xlin, ylin)
        axes[0].contourf(X, Y, Z, 20, cmap='RdGy')
        # axes[0].contourf(X, Y, Z, 20, cmap='RdGy')
        # show ground truth
        axes[1].contourf(X, Y, gmap, 20, cmap='RdGy')
        # axes[1].contouf(X, Y, gmap, 20, cmap='RdGy')
        # plt.imshow(Z)
        # plt.imshow(gmap > 0)
        plt.show()

