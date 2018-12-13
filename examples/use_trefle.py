from trefle_engine import TrefleFIS

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from trefle.fitness_functions.output_thresholder import round_to_cls
from trefle.trefle_classifier import TrefleClassifier


# @profile(sort="cumulative", filename="/tmp/pyfuge.profile")
def run():
    import numpy as np
    import random

    np.random.seed(6)
    random.seed(6)

    # Load dataset
    data = load_breast_cancer()
    # data = load_iris()

    # Organize our data
    y_names = data["target_names"]
    print("target names", y_names)
    y = data["target"]
    y = y.reshape(-1, 1)
    X_names = data["feature_names"]
    print("features names", X_names)
    X = data["data"]

    # X, y = make_classification(
    #     n_samples=1000, n_features=10,  n_informative=5, n_classes=2
    # )
    # y = y.reshape(-1, 1)

    # multi_class_y_col = np.random.randint(0, 4, size=y.shape)
    # regr_y_col = np.random.random(size=y.shape) * 100 + 20
    # y = np.hstack((y, multi_class_y_col, regr_y_col))
    # print(y)

    # Split our data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    def fit(y_true, y_pred):
        y_pred_thresholded = round_to_cls(y_pred, n_classes=2)
        fitness_val = accuracy_score(y_true, y_pred_thresholded)
        return fitness_val

    # Initialize our classifier
    clf = TrefleClassifier(
        n_rules=3,
        n_classes_per_cons=[2],
        default_cons=[1],
        n_max_vars_per_rule=3,
        n_generations=10,
        pop_size=100,
        n_labels_per_mf=3,
        verbose=True,
        dc_weight=1,
        # p_positions_per_lv=16,
        n_lv_per_ind_sp1=40,
        fitness_function=fit,
    )

    # Train our classifier
    model = clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    clf.print_best_fuzzy_system()
    tff_str = clf.get_best_fuzzy_system_as_tff()
    print(tff_str)

    yolo = TrefleFIS.from_tff(tff_str)
    yolo.describe()

    # fis = clf.get_best_fuzzy_system()
    # print("best fis is ", end="")
    # print(fis)

    # FISViewer(fis).show()

    # Evaluate accuracy
    print("Simple run score: ")

    y_pred_thresholded = round_to_cls(y_pred, n_classes=2)
    print("acc", accuracy_score(y_test, y_pred_thresholded))
    print(classification_report(y_test, y_pred_thresholded))


if __name__ == "__main__":
    run()
