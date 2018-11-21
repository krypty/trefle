import numpy as np
from numpy.testing import assert_array_equal

from trefle.fitness_functions.output_thresholder import round_to_cls


def test_distribution_between_binary_outputs_should_be_equal():
    raw_outputs = np.linspace(0, 1, 11)
    thresholded_outputs = round_to_cls(raw_outputs, n_classes=2)

    expected_array = 6 * [0] + 5 * [1]
    assert_array_equal(thresholded_outputs, expected_array)


def test_distribution_between_multiclass_output_should_be_equal():
    n_classes = 4
    raw_outputs = np.linspace(0, 1, 12) * (n_classes - 1)
    thresholded_outputs = round_to_cls(raw_outputs, n_classes=n_classes)
    expected_array = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]

    assert_array_equal(thresholded_outputs, expected_array)
