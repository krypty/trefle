from numpy.testing import assert_array_equal

from tests.fixture.trefle_classifier_test import (
    get_sample_data,
    get_trefle_classifier_instance,
)
from trefle.fitness_functions.output_thresholder import round_to_cls


def test_predict_classes_should_return_same_results_as_predict_plus_manual_round():
    X_train, X_test, y_train, y_test = get_sample_data()

    clf = get_trefle_classifier_instance(X_train, X_test, y_train, y_test)

    y_pred = clf.predict_X_test()

    y_pred_rounded = round_to_cls(y_pred, n_classes=3)

    y_pred_classes = clf.predict_X_test_classes()
    print(y_pred_classes)

    assert_array_equal(y_pred_classes, y_pred_rounded)
