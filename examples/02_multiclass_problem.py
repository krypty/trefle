import random

import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from trefle.fitness_functions.output_thresholder import round_to_cls
from trefle.trefle_classifier import TrefleClassifier


def main():
    np.random.seed(0)
    random.seed(0)

    # Load dataset
    data = load_iris()

    N_CLASSES = 3

    # Organize our data
    X = data["data"]
    y = data["target"]
    y = np.reshape(y, (-1, 1))  # output needs to be at least 1 column wide

    # Split our data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # Declare the fitness function we want to use
    def fit(y_true, y_pred):
        # y_pred are floats in [0, n_classes-1]. To use accuracy metric we need
        # to binarize the output using round_to_cls()
        y_pred_bin = round_to_cls(y_pred, N_CLASSES)
        return accuracy_score(y_true, y_pred_bin)

    # Initialize our classifier
    clf = TrefleClassifier(
        n_rules=2,
        n_classes_per_cons=[N_CLASSES],  # there is only 1
        #                                # consequent with 3 classes
        n_labels_per_mf=4,  # use 4 labels LOW, MEDIUM, HIGH, VERY HIGH
        default_cons=[1],  # default rule yield the class 1
        n_max_vars_per_rule=4,  # let's use the 4 iris variables (PL, PW, SL, SW)
        n_generations=30,
        fitness_function=fit,
        verbose=True,
    )

    # Train our classifier
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict_classes(X_test)

    clf.print_best_fuzzy_system()

    # Evaluate f1 score.
    # Important /!\ the fitness can be different than the scoring function
    score = f1_score(y_test, y_pred, average="weighted")
    print("Score on test set: {:.3f}".format(score))


if __name__ == "__main__":
    main()
