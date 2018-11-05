import random

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from pyfuge.evo.skfuge.trefle_classifier import TrefleClassifier


def get_sample_data():
    np.random.seed(6)
    random.seed(6)

    data = load_iris()

    y = data["target"]
    y = y.reshape(-1, 1)
    X = data["data"]

    # Split our data
    return train_test_split(X, y, test_size=0.33)


def get_trefle_classifier_instance(X_train, X_test, y_train, y_test):
    clf = TrefleClassifierTest(X_train, y_train, X_test, y_test)
    return clf


class TrefleClassifierTest:
    def __init__(self, X_train, y_train, X_test, y_test):
        self._X_train = X_train
        self._y_train = y_train
        self._X_test = X_test
        self._y_test = y_test

        self._clf = self._create_fis()

    def predict_X_test(self):
        return self._clf.predict(self._X_test)

    def get_best_fuzzy_system_as_tff(self):
        return self._clf.get_best_fuzzy_system_as_tff()

    def _create_fis(self):
        def fit(y_true, y_pred):
            rmse = -mean_squared_error(y_true, y_pred)
            return rmse

        # Initialize our classifier
        clf = TrefleClassifier(
            n_rules=3,
            n_classes_per_cons=[3],
            default_cons=[1],
            n_max_vars_per_rule=4,
            n_generations=5,
            pop_size=100,
            n_labels_per_mf=3,
            dc_weight=2,
            fitness_function=fit,
        )

        clf.fit(self._X_train, self._y_train)
        return clf
