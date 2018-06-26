from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

from pyfuge.evo.skfuge.scikit_fuge import FugeClassifier
from pyfuge.fs.view.fis_viewer import FISViewer


def gs():
    # Load dataset
    data = load_breast_cancer()

    # Organize our data
    y_names = data['target_names']
    y = data['target']
    X_names = data['feature_names']
    X = data['data']

    # Split our data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    tuned_params = {"dont_care_prob": [None, 0.1, 0.5, 0.8, 0.99]}

    gs = GridSearchCV(
        FugeClassifier(n_rules=3, n_generations=100, pop_size=100,
                       n_labels_per_mf=3, verbose=True),
        tuned_params)

    gs.fit(X_train, y_train)

    # print the best param value for dont_care_prob
    print(gs.best_params_)


def run_compare():
    # Load dataset
    # data = load_breast_cancer()
    data = load_iris()

    # Organize our data
    y_names = data['target_names']
    y = data['target']
    X_names = data['feature_names']
    X = data['data']

    # Split our data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    def classify(clf):
        # Initialize our classifier

        # Train our classifier
        model = clf.fit(X_train, y_train)

        # Make predictions
        preds = clf.predict(X_test)
        # print(preds)

        # Evaluate accuracy
        return accuracy_score(y_test, preds)

    fuge_win_count = 0
    n = 5
    for i in range(n):
        print("running battle {}...".format(i + 1))
        a = classify(FugeClassifier(n_rules=3, n_generations=130, pop_size=100))
        b = classify(KNeighborsClassifier())

        print("FUGE ({:.3f}) vs other ({:.3f})".format(a, b))
        fuge_win_count += int(a > b)
    print("")
    print("FUGE wins {}/{}".format(fuge_win_count, n))


def run():
    # Load dataset
    data = load_breast_cancer()
    # data = load_iris()

    # Organize our data
    y_names = data['target_names']
    y = data['target']
    X_names = data['feature_names']
    X = data['data']

    # FIXME support regression problems
    # X, y = load_boston(return_X_y=True)

    # Split our data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # Initialize our classifier
    clf = FugeClassifier(n_rules=2, n_generations=1000, pop_size=300,
                         n_labels_per_mf=3, dont_care_prob=0.9, verbose=True)

    # Train our classifier
    model = clf.fit(X_train, y_train)

    # Make predictions
    preds = clf.predict(X_test)

    fis = clf.get_best_fuzzy_system()

    print(fis)

    FISViewer(fis).show()

    # Evaluate accuracy
    print("Simple run score: ")
    print(accuracy_score(y_test, preds))


if __name__ == '__main__':
    # gs()
    run()
    # run_compare()
