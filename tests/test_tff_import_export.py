from pyfuge_c import TrefleFIS

import numpy as np

from tests.fixture.trefle_classifier_test import (
    get_sample_data,
    get_trefle_classifier_instance,
)


def test_exported_tff_should_give_the_same_predictions_as_the_system_it_is_based_on():
    X_train, X_test, y_train, y_test = get_sample_data()

    clf = get_trefle_classifier_instance(X_train, X_test, y_train, y_test)
    tff_str = clf.get_best_fuzzy_system_as_tff()
    trefle_fis = TrefleFIS.from_tff(tff_str)

    y_pred = clf.predict_X_test()
    y_pred_exported = trefle_fis.predict(X_test)

    assert np.isclose(
        y_pred, y_pred_exported
    ).all(), "results between TrefleFIS and SingletonFIS should be the same but are not"
