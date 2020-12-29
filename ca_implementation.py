from typing import Callable

import numpy as np


class CA:
    def __init__(
        self,
        n: int,
        state_dim: int,
        initialise_rule: Callable[[], None],
        update_rule: Callable[[np.ndarray], None],
        pad_size: int
    ):
        self.pad = pad_size
        self.n: int = n
        self.state_dim = state_dim
        self.initialise_rule: Callable = initialise_rule
        self.update_rule: Callable = update_rule
        self.grid: np.ndarray = np.zeros((n, n, state_dim))
        self.grid = np.pad(self.grid, ((self.pad, self.pad), (self.pad, self.pad), (0, 0)), constant_values=(-1, -1))
        self.initialise()

    def initialise(self):
        self.initialise_rule(self.grid[self.pad:self.n - self.pad, self.pad:self.n - self.pad])

    def _perceive(self, i: int, j: int) -> np.ndarray:
        i_start = max(0, i - 1)
        i_end = min(i + 2, self.n - 1)
        j_start = max(0, j - 1)
        j_end = min(j + 2, self.n - 1)
        x = self.grid[i_start:i_end, j_start:j_end]
        return x

    def update(self, sub_array: np.ndarray) -> np.ndarray:
        return self.update_rule(sub_array)

    def step(self):
        for i in range(self.pad, self.n - self.pad):
            for j in range(self.pad, self.n - self.pad):
                x = self._perceive(i, j)
                self.update(x)

    def reset(self):
        self.__init__(self.n, self.state_dim, self.initialise_rule, self.update_rule, self.pad)
