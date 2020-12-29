import pickle
from typing import Any

import numpy as np
from matplotlib import pyplot as plt


def display_2d_grid(grid: np.ndarray) -> None:
    plt.imshow(grid)
    plt.show()


def save_pickle(content: Any, path: str) -> None:
    with open(path, 'wb') as f:
        pickle.dump(content, f)


def load_pickle(path: str) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)
