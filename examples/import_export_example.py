import random
from trefle_engine import TrefleFIS

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from trefle.fitness_functions.output_thresholder import round_to_cls
from trefle.trefle_classifier import TrefleClassifier


def main():
    set_random_seed()

    X_test, X_train, y_test, y_train = get_dataset()
    clf = run_and_get_classifier(X_test, X_train, y_test, y_train)

    # tff (Trefle Fuzzy system File) is a json representation of the fuzzy system.
    # It is meant to be saved on disk or to be used by LFA Toolbox
    # (https://github.com/krypty/lfa_toolbox)
    tff = clf.get_best_fuzzy_system_as_tff()
    print(tff)

    # Export: save the fuzzy model to disk
    with open("my_saved_model.tff", mode="w") as f:
        f.write(tff)

    # Import from file
    fis = TrefleFIS.from_tff_file("my_saved_model.tff")

    # In the future, it could possible to call clf.predict_classes() directly
    # see issue #1
    y_pred_test = fis.predict(X_test)

    y_pred_test_bin = round_to_cls(y_pred_test, n_classes=2)
    print_score(y_pred_test_bin, y_test)

    # Import from string
    fis2 = TrefleFIS.from_tff(tff)
    y_pred_test = fis2.predict(X_test)

    y_pred_test_bin = round_to_cls(y_pred_test, n_classes=2)
    print_score(y_pred_test_bin, y_test)


def set_random_seed():
    np.random.seed(0)
    random.seed(0)


def get_dataset():
    # Load dataset
    data = load_breast_cancer()
    # Organize our data
    X = data["data"]
    print(X.shape)
    y = data["target"]
    y = np.reshape(y, (-1, 1))  # output needs to be at least 1 column wide
    # Split our data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    return X_test, X_train, y_test, y_train


def run_and_get_classifier(X_test, X_train, y_test, y_train):
    # Declare the fitness function we want to use
    def fit(y_true, y_pred):
        # y_pred are floats in [0, n_classes-1]. To use accuracy metric we need
        # to binarize the output using round_to_cls()
        y_pred_bin = round_to_cls(y_pred, n_classes=2)
        return accuracy_score(y_true, y_pred_bin)

    # Initialize our classifier
    clf = TrefleClassifier(
        n_rules=4,
        n_classes_per_cons=[2],  # there is only 1 consequent with 2 classes
        n_labels_per_mf=3,  # use 3 labels LOW, MEDIUM, HIGH
        default_cons=[0],  # default rule yield the class 0
        n_max_vars_per_rule=3,  # WBCD dataset has 30 variables, here we force
        # to use a maximum of 3 variables per rule
        # to have a better interpretability
        # In total we can have up to 3*4=12 different variables
        # for a fuzzy system
        n_generations=5,
        fitness_function=fit,
        verbose=True,
    )
    # Train our classifier
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict_classes(X_test)
    clf.print_best_fuzzy_system()

    print_score(y_pred, y_test)
    return clf


def print_score(y_pred, y_test):
    # Evaluate accuracy
    score = accuracy_score(y_test, y_pred)
    print("Score on test set: {:.3f}".format(score))


if __name__ == "__main__":
    main()
