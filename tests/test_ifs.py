import numpy as np

from evo.playground.ifs_without_evo import IFSUtils


def test_unitfloat2idx_equal_weights():
    """
    Test case 1: equals weights
    """
    float_li = [0.0, 0.5, 1.0]
    expected_indices = [0, 1, 2]
    weights = np.array([1, 1, 1])

    _check_unitfloat2idx(float_li, expected_indices, weights)


def test_unitfloat2idx_different_weights():
    """
    Test case 2: different weights
    """
    float_li = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    expected_indices = [0, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3]
    weights = np.array([1, 1, 1, 7])

    _check_unitfloat2idx(float_li, expected_indices, weights)


def test_unitfloat2idx_different_weights_less_than_one():
    """
    Test case 3: different weights < 1.0
    """
    float_li = [0.0, 0.25, 0.5, 1.0]
    expected_indices = [0, 1, 1, 1]
    weights = np.array([0.25, 1])

    _check_unitfloat2idx(float_li, expected_indices, weights)


def _check_unitfloat2idx(float_li, expected_indices, weights):
    for i in range(len(float_li)):
        idx = IFSUtils.unitfloat2idx(float_li[i], weights)
        print(float_li[i], idx)
        assert idx == expected_indices[i]

    # assert False


def test_evo_ants2ifs_ants():
    # floats given from an individual
    evo_ants = np.array([
        [0.0, 1.0],  # r0
        [0.5, 0.5],  # r1
        [0.25, 0.75],  # r2
    ])

    # expected output indices
    exp_arr = np.array([
        [0, 2],
        [2, 2],
        [1, 2],
    ])

    labels_names = ["L", "H", "DC"]
    labels_weights = np.ones(len(labels_names))

    ifs_ants = IFSUtils.evo_ants2ifs_ants(evo_ants, labels_weights)

    try:
        assert np.allclose(exp_arr, ifs_ants)
    except AssertionError:
        print("ERROR !")
        print(exp_arr)
        print("^exp---out--v")
        print(ifs_ants)


def test_evo_ants2ifs_ants_2():
    # floats given from an individual
    evo_ants = np.array([
        [0.0, 1.0],  # r0
        [0.5, 0.5],  # r1
        [0.25, 0.75],  # r2
    ])

    # expected output indices
    exp_arr = np.array([
        [0, 3],
        [2, 2],
        [1, 2],
    ])

    labels_weights = np.array([0.5, 0.5, 0.5, 2])
    ifs_ants = IFSUtils.evo_ants2ifs_ants(evo_ants, labels_weights)

    try:
        assert np.allclose(exp_arr, ifs_ants)
    except AssertionError:
        print("ERROR !")
        print(exp_arr)
        print("^exp---out--v")
        print(ifs_ants)
