import numpy  as np
import torch
import torch.utils.data

A = [
    [0, 0, 0, 1, 0, 0, 0],
    [0, 0, 1, 0, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 1, 0],
]

K = [
    [0, 1, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 1, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 1, 0],
]

letter_array_dict = {
    'A': A,
    'K': K
}


def letter_to_array(letter: str) -> np.ndarray:
    letter = letter.upper()
    if letter in letter_array_dict:
        return np.array(letter_array_dict[letter])
    else:
        raise ValueError(f"Can't convert letter '{letter}', try one of {set(letter_array_dict.keys())}")


def generate_training_data1(
    n_data_points: int,
    state_dim: int,
    grid_size: int,
    letter: str

):
    X = np.random.rand(n_data_points, state_dim, grid_size, grid_size).astype(float)
    # X[:, 0, :, :] = 0
    # X[:, 0, 3:, 3:] = 1
    X = torch.from_numpy(X).float()
    y = np.array([letter_to_array(letter) for _ in range(n_data_points)]).astype(float)
    y = torch.from_numpy(y).float()

    return X, y


def generate_training_data2(
    n_data_points: int,
    state_dim: int,
    grid_size: int,
    n_letters: int,
    letter: str
):
    X = np.random.rand(n_data_points, state_dim, grid_size, grid_size).astype(float)
    top_grid = X[:, 0]
    top_grid = top_grid.reshape((n_data_points, -1))
    top_grid[:] = 0
    col_ix = np.array([np.random.choice(top_grid.shape[-1], n_letters, replace=False) for _ in range(n_data_points)])
    row_ix, _ = np.indices(col_ix.shape)
    top_grid[row_ix, col_ix] = 1
    top_grid = top_grid.reshape((n_data_points, grid_size, grid_size))
    X[:, 0] = top_grid

    X = torch.from_numpy(X).float()
    y = np.array([letter_to_array(letter) for _ in range(n_data_points)]).astype(float)
    y = torch.from_numpy(y).float()

    return X, y
