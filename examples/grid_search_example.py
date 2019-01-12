import random

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, make_scorer, recall_score, \
    precision_score
from sklearn.model_selection import train_test_split, GridSearchCV

from trefle.fitness_functions.output_thresholder import round_to_cls
from trefle.trefle_classifier import TrefleClassifier


def main():
    """
    Executing this method is time consuming
    """

    np.random.seed(0)
    random.seed(0)

    # Load dataset
    data = load_breast_cancer()

    # Organize our data
    X = data["data"]
    print(X.shape)
    y = data["target"]
    y = np.reshape(y, (-1, 1))  # output needs to be at least 1 column wide

    # Split our data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # Declare the fitness function we want to use
    def evaluate(y_true, y_pred):
        # y_pred are floats in [0, n_classes-1]. To use accuracy metric we need
        # to binarize the output using round_to_cls()
        y_pred_bin = round_to_cls(y_pred, n_classes=2)
        return accuracy_score(y_true, y_pred_bin)

    # Initialize our classifier
    # note that some arguments are mandatory such as n_rules. If you want to
    # grid search that parameter you must set it to None before tuning it.
    estimator = TrefleClassifier(
        n_rules=None,  # mandatory argument, set to None and tune it below
        n_classes_per_cons=[2],  # there is only 1 consequent with 2 classes
        n_labels_per_mf=3,  # use 3 labels LOW, MEDIUM, HIGH
        default_cons=None,
        n_max_vars_per_rule=3,  # WBCD dataset has 30 variables, here we force
        # to use a maximum of 3 variables per rule
        # to have a better interpretability
        # In total we can have up to 3*4=12 different variables
        # for a fuzzy system
        n_generations=20,
        fitness_function=evaluate,
        verbose=True,
    )

    def get_fitness_functions():
        def get_recall_and_precision_score(y_true, y_pred):
            y_pred_bin = round_to_cls(y_pred, n_classes=2)
            recall = recall_score(y_true, y_pred_bin)
            precision = precision_score(y_true, y_pred_bin)
            return recall, precision

        def ff1(y_true, y_pred):
            recall, precision = get_recall_and_precision_score(y_true, y_pred)
            return (1.0 * recall + 2.0 * precision) / 3.0

        def ff2(y_true, y_pred):
            recall, precision = get_recall_and_precision_score(y_true, y_pred)
            return (2.0 * recall + 1.0 * precision) / 3.0

        def ff3(y_true, y_pred):
            recall, precision = get_recall_and_precision_score(y_true, y_pred)
            return (1.0 * recall + 3.0 * precision) / 4.0

        return ff1, ff2, ff3

    tuned_parameters = [
        {"n_rules": [2, 5], "default_cons": [[0], [1]]},
        {
            "n_rules": [4, 5],
            "default_cons": [[0], [1]],
            "n_max_vars_per_rule": [3, 5, 6],
        },
        {
            "n_rules": [4],
            "default_cons": [[0]],
            "fitness_function": get_fitness_functions(),
        },
    ]

    # Note that the scoring reuses the evaluate function but fitness_function
    # (i.e. the function that compares models for a given configuration/run)
    # can be different than scoring function (i.e. the function that compares
    # best individuals between different configurations/runs)
    clf = GridSearchCV(estimator, tuned_parameters, cv=3, scoring=make_scorer(evaluate))

    # Train our classifier
    clf.fit(X_train, y_train)

    print("Best params: ")
    print(clf.best_params_)

    best_estimator = clf.best_estimator_
    y_pred_test = best_estimator.predict_classes(X_test)
    print(accuracy_score(y_true=y_test, y_pred=y_pred_test))
    best_estimator.print_best_fuzzy_system()


if __name__ == "__main__":
    print("[warning] this example can take a while to run")
    main()
