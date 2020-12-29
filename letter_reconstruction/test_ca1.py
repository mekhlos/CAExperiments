from pathlib import Path

import numpy as np

import utils
from data.data_dir import data_dir_path


def apply_conv(X: np.ndarray, weights: np.ndarray, bias: np.ndarray):
    X_new = X.copy()

    for i in range(1, X.shape[-1] - 1):
        for j in range(1, X.shape[-2] - 1):
            sub_X = X[:, :, i - 1:i + 2, j - 1:j + 2]
            s = np.einsum('bixy,oixy->bo', sub_X, weights) + bias
            X_new[:, :, i, j] = s

    X_new[X_new < 0] = 0

    return X_new


def run_simulation(X: np.ndarray, weights: np.ndarray, bias: np.ndarray):
    X = np.pad(X, ((0, 0), (0, 0), (1, 1), (1, 1)), constant_values=0)

    for _ in range(50):
        X = apply_conv(X, weights, bias)

        print('\n', X[0, 0].round(0))
        utils.display_2d_grid(X[0, 0])

    return X[:, :, 1:-1, 1:-1]


if __name__ == '__main__':
    p = sorted(Path(f'{data_dir_path}').glob('*.pickle'))[-1]
    print(p)
    params = utils.load_pickle(p)
    w = params['weight']
    b = params['bias']

    initial_state = np.ones((1, 4, 7, 7))
    print(initial_state[0, 0].round(1))
    final_state = run_simulation(initial_state, w, b)
    print(final_state[0, 0].round(0))
