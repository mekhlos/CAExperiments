import torch.utils.data


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, item):
        return self.X[item], self.y[item]

    def __len__(self):
        return len(self.X)
