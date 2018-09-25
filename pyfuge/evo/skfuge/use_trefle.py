import numpy as np
from pyfuge.evo.skfuge.trefle_classifier import TrefleClassifier
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV

from pyfuge.evo.skfuge.fitness_functions import basic_fitness_functions
from pyfuge.evo.skfuge.scikit_fuge import FugeClassifier
from pyfuge.fs.view.fis_viewer import FISViewer


# def gs():
#     # Load dataset
#     data = load_breast_cancer()
#
#     # Organize our data
#     y_names = data['target_names']
#     y = data['target']
#     X_names = data['feature_names']
#     X = data['data']
#
#     # Split our data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
#
#     tuned_params = {"dont_care_prob": [None, 0.1, 0.5, 0.8, 0.99]}
#
#     gs = GridSearchCV(
#         FugeClassifier(n_rules=3, n_generations=100, pop_size=100,
#                        n_labels_per_mf=3, verbose=True),
#         tuned_params)
#
#     gs.fit(X_train, y_train)
#
#     # print the best param value for dont_care_prob
#     print(gs.best_params_)


# def run_compare():
#     # Load dataset
#     # data = load_breast_cancer()
#     data = load_iris()
#
#     # Organize our data
#     y_names = data['target_names']
#     y = data['target']
#     X_names = data['feature_names']
#     X = data['data']
#
#     # Split our data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
#
#     def classify(clf):
#         # Initialize our classifier
#
#         # Train our classifier
#         model = clf.fit(X_train, y_train)
#
#         # Make predictions
#         preds = clf.predict(X_test)
#         # print(preds)
#
#         # Evaluate accuracy
#         return accuracy_score(y_test, preds)
#
#     fuge_win_count = 0
#     n = 5
#     for i in range(n):
#         print("running battle {}...".format(i + 1))
#         a = classify(FugeClassifier(n_rules=3, n_generations=130, pop_size=100))
#         b = classify(KNeighborsClassifier())
#
#         print("FUGE ({:.3f}) vs other ({:.3f})".format(a, b))
#         fuge_win_count += int(a > b)
#     print("")
#     print("FUGE wins {}/{}".format(fuge_win_count, n))


def run():
    import numpy as np
    import random

    np.random.seed(6)
    random.seed(6)

    # Load dataset
    # data = load_breast_cancer()
    data = load_iris()

    # Organize our data
    y_names = data["target_names"]
    y = data["target"]
    y = y.reshape(-1, 1)
    X_names = data["feature_names"]
    X = data["data"]

    # FIXME support regression problems
    # X, y = load_boston(return_X_y=True)

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
        y_pred_thresholded = round_to_cls(y_pred, n_classes=3)
        fitness_val = accuracy_score(y_true, y_pred_thresholded)
        rmse = 1.0 - mean_squared_error(y_true, y_pred)
        return rmse + fitness_val

    # Initialize our classifier
    clf = TrefleClassifier(
        n_rules=3,
        n_classes_per_cons=[3],
        default_cons=[2],
        n_max_vars_per_rule=4,
        n_generations=100,
        pop_size=80,
        n_labels_per_mf=3,
        verbose=True,
        dc_weight=1,
        p_positions_per_lv=512,
        n_lv_per_ind_sp1=30,
        fitness_function=fit,
    )

    # Train our classifier
    model = clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # fis = clf.get_best_fuzzy_system()
    clf.get_best_fuzzy_system()

    # print(fis)

    # FISViewer(fis).show()

    # Evaluate accuracy
    print("Simple run score: ")

    y_pred_thresholded = round_to_cls(y_pred, n_classes=3)
    print("acc", accuracy_score(y_test, y_pred_thresholded))
    # print(classification_report(y_test, y_pred))


def run2():
    # Load dataset
    data = load_breast_cancer()

    # Organize our data
    y_names = data["target_names"]
    y = data["target"]
    X_names = data["feature_names"]
    X = data["data"]

    # Split our data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    fit_func = basic_fitness_functions.weighted_binary_classif_metrics(
        f1_w=1, acc_w=1, mse_w=0.1
    )
    clf = FugeClassifier(
        n_rules=3,
        n_generations=100,
        pop_size=100,
        dont_care_prob=0.9,
        n_labels_per_mf=3,
        fitness_function=fit_func,
        verbose=True,
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    from sklearn.metrics import f1_score

    score = f1_score(y_true=y_test, y_pred=y_pred)
    print("score", score)
    score = accuracy_score(y_true=y_test, y_pred=y_pred)
    print("score", score)

    fis = clf.get_best_fuzzy_system()
    FISViewer(fis).show()


def test_custom_fit_functions_with_gs():
    fit_funcs = basic_fitness_functions.weighted_binary_classif_metrics(
        sen_w=np.linspace(0, 1, 2), spe_w=np.linspace(0, 1, 3)
    )
    print("n of fit funcs", len(fit_funcs))
    print(fit_funcs)

    # Load dataset
    data = load_breast_cancer()

    # Organize our data
    y_names = data["target_names"]
    y = data["target"]
    X_names = data["feature_names"]
    X = data["data"]

    # Split our data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    tuned_params = {"fitness_function": fit_funcs}

    # do not use n_jobs > 1
    from sklearn.metrics import make_scorer
    from sklearn.metrics import f1_score

    gs = GridSearchCV(
        FugeClassifier(
            n_rules=3,
            n_generations=50,
            pop_size=50,
            n_labels_per_mf=3,
            dont_care_prob=0.9,
            verbose=True,
        ),
        tuned_params,
        scoring=make_scorer(f1_score),
    )  # optional

    gs.fit(X_train, y_train)

    # print the best param value for fitness_function arg
    print(gs.best_params_)
    best_ff = gs.best_params_["fitness_function"]
    print(best_ff)
    print("best sen_w", best_ff["sen_w"])
    print("best spe_w", best_ff["spe_w"])


if __name__ == "__main__":
    # gs()
    run()
    # run_compare()
    # test_custom_fit_functions_with_gs()
    # run2()
