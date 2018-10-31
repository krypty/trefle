from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.model_selection import train_test_split

from pyfuge.evo.helpers.fuzzy_labels import Label3
from pyfuge.evo.skfuge.trefle_classifier import TrefleClassifier


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

    # FIXME support regression problems
    # X, y = load_boston(return_X_y=True)

    # X, y = make_classification(
    #     n_samples=1000, n_features=10,  n_informative=5, n_classes=2
    # )
    # y = y.reshape(-1, 1)

    multi_class_y_col = np.random.randint(0, 4, size=y.shape)
    regr_y_col = np.random.random(size=y.shape) * 100 + 20
    y = np.hstack((y, multi_class_y_col, regr_y_col))
    # print(y)

    # Split our data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # X_train = X_train[:3]
    # y_train = y_train[:3]

    def round_to_cls(arr, n_classes):
        bins = np.linspace(0, n_classes - 1, n_classes + 1)
        # print(bins)
        c = np.searchsorted(bins, arr)
        c -= 1
        c = np.clip(c, 0, n_classes - 1)
        return c

    # def fit(y_true, y_pred):
    #     return (1 - mean_squared_error(y_true, y_pred)) + 3.0*accuracy_score(
    #         y_true, round_to_cls(y_pred, n_classes=2)
    #     )

    def fit(y_true, y_pred):
        # y_pred_thresholded = round_to_cls(y_pred, n_classes=3)
        # fitness_val = accuracy_score(y_true, y_pred_thresholded)
        rmse = -mean_squared_error(y_true, y_pred)
        return rmse
        # return rmse + fitness_val
        # return fitness_val

    # Initialize our classifier
    clf = TrefleClassifier(
        n_rules=3,
        n_classes_per_cons=[2, 4, 0],
        default_cons=[1,2,Label3.MEDIUM],
        n_max_vars_per_rule=4,
        n_generations=10,
        pop_size=100,
        n_labels_per_mf=3,
        verbose=True,
        dc_weight=2,
        # p_positions_per_lv=16,
        n_lv_per_ind_sp1=40,
        fitness_function=fit,
    )

    # Train our classifier
    model = clf.fit(X_train, y_train)

    # Make predictions
    np.save("/tmp/X_test", X_test)
    y_pred = clf.predict(X_test)
    np.save("/tmp/y_pred", y_pred)

    # fis = clf.get_best_fuzzy_system()
    tff_str = clf.get_best_fuzzy_system_as_tff()
    open("/tmp/temp.tff", mode="w").write(tff_str)

    fis = clf.get_best_fuzzy_system()
    print("best fis is ", end="")
    print(fis)

    # y_pred_legacy = []
    # for crisp_values in X_test:
    #     crisp_values = {str(i): v for i, v in enumerate(crisp_values)}
    #     y_pred_legacy.append(fis.predict(crisp_values)["out0"])
    # y_pred_legacy = np.asarray(y_pred_legacy).reshape(-1, 1)
    #
    # # np.set_printoptions(suppress=True, precision=6)
    # print("y_pred vs y_pred_legacy")
    # print(y_pred.shape, y_pred_legacy.shape)
    # print(y_pred_legacy - y_pred)
    # print(np.isclose(y_pred, y_pred_legacy).all())


    # FISViewer(fis).show()

    # Evaluate accuracy
    print("Simple run score: ")

    # y_pred_thresholded = round_to_cls(y_pred, n_classes=3)
    # print("acc", accuracy_score(y_test, y_pred_thresholded))
    # print(classification_report(y_test, y_pred))
    print(mean_squared_error(y_test, y_pred, multioutput="raw_values"))


if __name__ == "__main__":
    run()
