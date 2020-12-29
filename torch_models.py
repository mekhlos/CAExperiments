import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, d_in, *d_layers, d_out):
        super().__init__()
        self.fc_in = nn.Linear(d_in, d_layers[0])
        self.fc_hidden = [nn.Linear(d1, d2) for d1, d2 in zip(d_layers, d_layers[1:])]
        self.fc_out = nn.Linear(d_layers[-1], d_out)

    def forward(self, x):
        x = F.relu(self.fc_in(x))
        for fc in self.fc_hidden:
            x = F.relu(fc(x))
        x = self.fc_out(x)
        return x


class CNN(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.conv = nn.Conv2d(state_dim, state_dim, 3)

    def forward(self, x):
        x = F.relu(self.conv(x))
        return x
