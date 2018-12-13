import random

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from trefle.fitness_functions.output_thresholder import round_to_cls
from trefle.trefle_classifier import TrefleClassifier


def create_one_hot_from_array(y):
    # source: https://stackoverflow.com/a/29831596
    y_one_hot = np.zeros((y.size, y.max() + 1))
    y_one_hot[np.arange(y.size), y] = 1
    return y_one_hot


def main():
    np.random.seed(0)
    random.seed(0)

    # Load dataset
    data = load_iris()

    # Organize our data
    X = data["data"]
    y = data["target"]  # y.shape is (150,)
    y = create_one_hot_from_array(y)  # y.shape is now (150,3)

    # Split our data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # Declare the fitness function we want to use
    def fit(y_true, y_pred):
        # y_pred are floats in [0, n_classes-1]. To use accuracy metric we need
        # to binarize the output using round_to_cls()
        # Warning /!\ here since it has been one-hot-encoded we need to set
        # n_classes=2 instead n_classes=N_CLASSES because each consequent
        # is a binary class
        y_pred_bin = round_to_cls(y_pred, n_classes=2)
        return accuracy_score(y_true, y_pred_bin)

    # Initialize our classifier
    clf = TrefleClassifier(
        n_rules=3,  # here we need to increase the number of rule to 3
        #           # because we need at least 1 rule per class in the case
        #           # of a one-hot-encoded problem
        n_classes_per_cons=[2, 2, 2],  # there are 3 consequents with 2 classes
        #                              # each.
        n_labels_per_mf=4,  # use 4 labels LOW, MEDIUM, HIGH, VERY HIGH
        default_cons=[0, 0, 1],  # default rule yield the class 2
        n_max_vars_per_rule=4,  # let's use the 4 iris variables (PL, PW, SL, SW)
        n_generations=30,
        fitness_function=fit,
        verbose=True,
    )

    # Train our classifier
    clf.fit(X_train, y_train)

    # Make predictions
    # y_pred = clf.predict_classes(X_test)
    y_pred_raw = clf.predict(X_test)
    y_pred = round_to_cls(y_pred_raw, n_classes=2)

    clf.print_best_fuzzy_system()

    # Evaluate accuracy
    # Important /!\ the fitness can be different than the scoring function
    score = accuracy_score(y_test, y_pred)
    print("Score on test set: {:.3f}".format(score))


if __name__ == "__main__":
    main()
