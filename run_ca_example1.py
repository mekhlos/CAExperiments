import time

import matplotlib.pyplot as plt
import numpy as np

plt.ion()


def initialise(grid: np.ndarray) -> None:
    x = (np.random.rand(*grid.shape) >= 0.5).astype(int)
    grid[:] = x[:]


n = 30
grid = np.zeros((n, n))

initialise(grid)

n_steps = 100

for step in range(n_steps):
    new_grid = grid.copy()
    for i in range(n):
        for j in range(n):
            x_start = max(0, i - 1)
            x_end = min(n - 1, i + 2)
            y_start = max(0, j - 1)
            y_end = min(n - 1, j + 2)
            square = grid[x_start:x_end, y_start:y_end]
            n_neighbours = square.sum() - grid[i, j]
            if grid[i, j] == 0 and n_neighbours == 3:
                new_grid[i, j] = 1
            elif grid[i, j] == 1 and n_neighbours < 2 or n_neighbours > 3:
                new_grid[i, j] = 0

    grid = new_grid

    plt.imshow(grid)
    plt.show()
    time.sleep(1)
