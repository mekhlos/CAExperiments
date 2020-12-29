import torch
from torch.optim import Optimizer


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        device: str,
        criterion,
        optimizer: Optimizer
    ):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer

    def step(self, X, y_true):
        self.optimizer.zero_grad()
        y_pred = self.model(X)
        loss = self.criterion(y_pred, y_true)
        loss.backward()
        self.optimizer.step()

        return loss

    def train(self, train_loader, n_epochs: int):
        n_data_points = len(train_loader.dataset)
        n_batches = len(train_loader)

        for i in range(n_epochs):
            epoch_loss = 0
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)

                loss = self.step(data, target)
                epoch_loss += loss.item()

            print(
                f'Train Epoch: {i} ({i / n_epochs:.0%}) '
                f'Loss: {epoch_loss:.6f}'
            )

    def predict(self, X) -> float:
        return self.model(X)
