import numpy as np


def round_to_cls(arr, n_classes):
    bins = np.linspace(0, n_classes - 1, n_classes + 1)
    c = np.searchsorted(bins, arr)
    c -= 1
    c = np.clip(c, 0, n_classes - 1)
    return c
