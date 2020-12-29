from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from letter_reconstruction.data_generator import generate_training_data1
from letter_reconstruction.param_saver import save_params
from torch_utils import MyDataset
from torch_trainer import Trainer


class CNN(nn.Module):
    def __init__(self, state_dim: int, min_step: int, max_step: int):
        super().__init__()
        self.min_step = min_step
        self.max_step = max_step
        self.conv = nn.Conv2d(state_dim, state_dim, 3, padding=(1, 1))

    def forward(self, X):
        n_steps = np.random.randint(self.min_step, self.max_step)
        for i in range(n_steps):
            X = F.relu(self.conv(X))
        return X


@dataclass
class Hyperparams:
    state_dim: int


def get_customised_criterion(criterion):
    def f(y_pred, y_true):
        y_pred = y_pred[:, 0]
        return criterion(y_pred, y_true)

    return f


if __name__ == '__main__':
    hyperparams = Hyperparams(
        state_dim=4
    )

    X, y = generate_training_data1(
        n_data_points=100,
        state_dim=hyperparams.state_dim,
        grid_size=7,
        letter="k"
    )

    dataset = MyDataset(X, y)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=4)

    net = CNN(
        state_dim=hyperparams.state_dim,
        min_step=20,
        max_step=40
    )

    criterion = nn.MSELoss()
    customised_criterion = get_customised_criterion(criterion)
    optimizer = optim.Adam(net.parameters())

    trainer = Trainer(
        model=net,
        device="cpu",
        criterion=customised_criterion,
        optimizer=optimizer
    )

    trainer.train(train_loader, n_epochs=35)

    save_params(net)
