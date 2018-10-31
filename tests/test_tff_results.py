import numpy as np

from tests.fixture.trefle_classifier_test import (
    get_trefle_classifier_instance,
    get_sample_data,
)


def legacy_predict(X_test, fis):
    y_pred_legacy = []
    for crisp_values in X_test:
        crisp_values = {str(i): v for i, v in enumerate(crisp_values)}
        y_pred_legacy.append(fis.predict(crisp_values)["out0"])
    y_pred_legacy = np.asarray(y_pred_legacy).reshape(-1, 1)
    return y_pred_legacy


def test_binary_problem_predictions_should_be_the_same_between_singleton_fis_and_trefle_fis():
    X_train, X_test, y_train, y_test = get_sample_data()
    clf = get_trefle_classifier_instance(X_train, X_test, y_train, y_test)

    fis = clf.get_best_fuzzy_system()

    y_pred = clf.predict_X_test()
    y_pred_legacy = legacy_predict(X_test, fis)

    assert np.isclose(
        y_pred, y_pred_legacy
    ).all(), "results between TrefleFIS and SingletonFIS should be the same but are not"
