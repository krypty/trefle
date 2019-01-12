import random

import numpy as np
from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from trefle.evo.helpers.fuzzy_labels import Label4, Label6, Label5, Label3
from trefle.trefle_classifier import TrefleClassifier


def main():
    np.random.seed(0)
    random.seed(0)

    # Load dataset
    data = load_boston()

    # Organize our data
    X = data["data"]
    print(X.shape)
    y = data["target"]
    y = np.reshape(y, (-1, 1))  # output needs to be at least 1 column wide

    # Split our data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # Declare the fitness function we want to use
    def fit(y_true, y_pred):
        # Here no need to threshold y_pred because we are using a regression
        # metric
        return -mean_squared_error(y_true, y_pred)

    # Initialize our classifier
    clf = TrefleClassifier(
        n_rules=5,
        n_classes_per_cons=[0],  # In regression, there is no class (i.e. 0)
        n_labels_per_cons=Label4,  # use 4 labels LOW, MEDIUM, HIGH, VERY HIGH
        #                          # for the consequent
        #                          # Recall: even for continuous variables
        #                          # we use a label e.g.
        #                          # "[...] THEN temperature is LOW"
        n_labels_per_mf=2,  # use 2 labels LOW, HIGH (for the antecedents)
        default_cons=[Label4.VERY_HIGH()],  # default rule yield the 4th (and last) label
        n_max_vars_per_rule=2,
        n_generations=30,
        fitness_function=fit,
        verbose=True,
    )

    # Train our classifier
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict_classes(X_test)

    # Alternatively, you can use predict() which return non-thresholded y_pred
    # but you could need to add a threshold yourself. For example:
    #   y_pred_raw = clf.predict(X_test)
    #   y_pred = round_to_cls(y_pred_raw, n_classes=2)

    clf.print_best_fuzzy_system()

    # Evaluate accuracy
    score = mean_squared_error(y_test, y_pred)
    print("Score on test set: {:.3f}".format(score))


if __name__ == "__main__":
    main()
