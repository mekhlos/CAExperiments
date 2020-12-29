import time

import matplotlib.pyplot as plt
import numpy as np

from ca_implementation import CA
from utils import display_2d_grid


def ca1():
    def update(sub_grid: np.ndarray) -> None:
        mid_x, mid_y = sub_grid.shape[0] // 2, sub_grid.shape[1] // 2
        top_layer = sub_grid[:, :, 0]
        v = top_layer[mid_x, mid_y]
        n_neighbours = top_layer.sum() - v
        if v == 0 and n_neighbours == 3:
            sub_grid[mid_x, mid_y] = 1
        elif v == 1 and n_neighbours < 2 or n_neighbours > 3:
            sub_grid[mid_x, mid_y] = 0

    def initialise(grid: np.ndarray) -> None:
        x = (np.random.rand(*grid.shape) >= 0.7).astype(int)
        grid[:] = x[:]

    ca = CA(30, 1, initialise, update, 0)
    grid_2d = ca.grid[:, :, 0]
    display_2d_grid(grid_2d)
    time.sleep(1)

    for i in range(100):
        ca.step()
        grid_2d = ca.grid[:, :, 0]
        display_2d_grid(grid_2d)
        time.sleep(1)


if __name__ == '__main__':
    plt.ion()
    ca1()
