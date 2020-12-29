from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

from letter_reconstruction.data_generator import generate_training_data2
from letter_reconstruction.param_saver import save_params
from torch_utils import MyDataset
from torch_trainer import Trainer


class CNN(nn.Module):
    def __init__(self, state_dim: int, min_step: int, max_step: int):
        super().__init__()
        self.min_step = min_step
        self.max_step = max_step
        self.conv = nn.Conv2d(state_dim, state_dim, 3, padding=(1, 1), bias=False)

    def forward(self, X):
        X_history = []
        n_steps = np.random.randint(self.min_step, self.max_step)
        for i in range(n_steps):
            X = F.relu(self.conv(X))
            X[:, 0] = X[:, 0].round()
            X_history.append(X)
        return X_history


@dataclass
class Hyperparams:
    state_dim: int
    n_letters: int


def get_customised_criterion(criterion, n_letters):
    def f(X_history, y_true):
        loss1 = torch.tensor(0).float()
        for X in X_history:
            l = (torch.einsum('bxy->b', X[:, 0]) - n_letters) ** 2 / sum(X.shape)
            loss1 += l.sum().float()

        return criterion(X_history[-1][:, 0], y_true)  # + loss1 / len(X_history)

    return f


if __name__ == '__main__':
    hyperparams = Hyperparams(
        state_dim=8,
        n_letters=14
    )

    X, y = generate_training_data2(
        letter="k",
        n_data_points=100,
        state_dim=hyperparams.state_dim,
        grid_size=7,
        n_letters=hyperparams.n_letters
    )

    dataset = MyDataset(X, y)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=4)

    net = CNN(
        state_dim=hyperparams.state_dim,
        min_step=20,
        max_step=40
    )

    criterion = nn.MSELoss()
    customised_criterion = get_customised_criterion(criterion, hyperparams.n_letters)
    optimizer = optim.Adam(net.parameters())

    trainer = Trainer(
        model=net,
        device="cpu",
        criterion=customised_criterion,
        optimizer=optimizer
    )

    trainer.train(train_loader, n_epochs=35)

    save_params(net)
