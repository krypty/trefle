import inspect

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB

from pyfuge.evo.dataset.pf_dataset import PFDataset
from pyfuge.evo.experiment.base.simple_experiment import SimpleEAExperiment
from pyfuge.evo.experiment.simple_fis_individual import SimpleFISIndividual
from pyfuge.evo.fitness_evaluator.pyfuge_fitness_evaluator import \
    PyFUGEFitnessEvaluator
from pyfuge.evo.helpers.ind_evaluator_utils import IndEvaluatorUtils
# this will raise KeyError if invalid
from pyfuge.evo.helpers.native_ind_evaluator import NativeIndEvaluator
from pyfuge.fs.view.fis_viewer import FISViewer

labels_str_dict = {
    2: ("LOW", "HIGH", "DONT CARE"),
    3: ("LOW", "MEDIUM", "HIGH", "DONT CARE"),
    4: ("LOW", "MEDIUM", "HIGH", "VERY HIGH", "DONT CARE"),
    5: ("VERY LOW", "LOW", "MEDIUM", "HIGH", "VERY HIGH", "DONT CARE"),
}


class FugeClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, n_rules=3, n_labels_per_mf=2, pop_size=80,
                 n_generations=100, halloffame=3, dont_care_prob=None,
                 verbose=False):
        # assign self.XXX = XXX for all args
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

    def _setup(self):
        if self.n_labels_per_mf <= 1:
            raise ValueError("n_labels_per_mf must be > 1")

        if self.dont_care_prob is None:
            self.dont_care_prob = 1.0 / (1 + self.n_labels_per_mf)
        elif not (0.0 < self.dont_care_prob < 1.0):
            raise ValueError("dont_care_prob must be between 0.0 and 1.0")

    def fit(self, X, y):
        self._setup()

        self._ds = PFDataset(X, y)
        self._n_vars = self._ds.N_VARS

        # TODO: allow the user to choose the default rule
        self._default_rule_output = self._ds.y[0].tolist()

        self._mf_label_names = labels_str_dict[self.n_labels_per_mf]

        self._labels_weights = self.compute_labels_weights(self.dont_care_prob,
                                                           self.n_labels_per_mf)

        # Setup the individuals structure
        self._fis_ind = SimpleFISIndividual(
            n_vars=self._n_vars,
            n_rules=self.n_rules,
            n_max_var_per_rule=self._n_vars,  # TODO remove me
            mf_label_names=self._mf_label_names,
            default_rule_output=self._default_rule_output,
            dataset=self._ds,
            labels_weights=self._labels_weights
        )

        # Setup and run the experiment
        exp = SimpleEAExperiment(
            dataset=self._ds,
            fis_individual=self._fis_ind,
            fitevaluator=PyFUGEFitnessEvaluator(),
            N_POP=self.pop_size,
            N_GEN=self.n_generations,
            HOF=self.halloffame,
            verbose=self.verbose
        )

        self._best_ind = exp.get_top_n()[0]
        return self

    def predict(self, X):
        self._ensure_fit()

        var_ranges = IndEvaluatorUtils.compute_vars_range(X)

        ind_evaluator = NativeIndEvaluator(
            ind_n=len(self._best_ind),
            observations=X,
            n_rules=self.n_rules,
            max_vars_per_rule=self._n_vars,  # TODO remove me
            n_labels=len(self._mf_label_names),
            n_consequents=len(self._default_rule_output),
            default_rule_cons=np.array(self._default_rule_output),
            vars_ranges=var_ranges,
            labels_weights=self._labels_weights
        )

        self._y_pred = ind_evaluator.predict_native(self._best_ind)

        if self._y_pred.shape[0] > 1:
            return np.argmax(self._y_pred, axis=1)
        else:
            return np.round(self._y_pred)

    def get_last_unthresholded_predictions(self):
        self._ensure_fit()

        return self._y_pred

    def get_best_fuzzy_system(self):
        self._ensure_fit()

        return self._fis_ind.convert_to_fis(self._best_ind)

    def _ensure_fit(self):
        if getattr(self, "_fis_ind") is None:
            raise ValueError("You must use fit() first")

    @staticmethod
    def compute_labels_weights(dont_care_prob, n_labels):
        other_labels_prob = 1.0 - dont_care_prob
        # +1 because sum(prob) = dc_prob + others_prob
        weights = [other_labels_prob / (n_labels + 1)] * n_labels
        weights.append(dont_care_prob)
        return np.array(weights)


def main():
    # Load dataset
    data = load_breast_cancer()
    # data = load_iris()

    # Organize our data
    label_names = data['target_names']
    labels = data['target']
    feature_names = data['feature_names']
    features = data['data']

    def cls_a():
        # Split our data
        train, test, train_labels, test_labels = train_test_split(features,
                                                                  labels,
                                                                  test_size=0.33)
        # Initialize our classifier
        # gnb = GaussianNB()
        gnb = FugeClassifier(n_rules=3, n_generations=200, pop_size=100)

        # Train our classifier
        model = gnb.fit(train, train_labels)

        # Make predictions
        preds = gnb.predict(test)
        # print(preds)

        # Evaluate accuracy
        return accuracy_score(test_labels, preds)

    def cls_b():
        # Split our data
        train, test, train_labels, test_labels = train_test_split(features,
                                                                  labels,
                                                                  test_size=0.33)
        # Initialize our classifier
        gnb = GaussianNB()
        # gnb = FugeClassifier()

        # Train our classifier
        model = gnb.fit(train, train_labels)

        # Make predictions
        preds = gnb.predict(test)
        # print(preds)

        # Evaluate accuracy
        return accuracy_score(test_labels, preds)

    # fuge_win_count = 0
    # for _ in range(10):
    #     a = cls_a()
    #     b = cls_b()
    #
    #     fuge_win_count += int(a > b)
    # print("FUGE wins {}/10".format(fuge_win_count))

    # Split our data
    train, test, train_labels, test_labels = train_test_split(features,
                                                              labels,
                                                              test_size=0.33)
    # Initialize our classifier
    # gnb = GaussianNB()
    clf = FugeClassifier(n_rules=3, n_generations=200, pop_size=100,
                         n_labels_per_mf=2, dont_care_prob=0.95, verbose=True)

    # Train our classifier
    model = clf.fit(train, train_labels)

    # Make predictions
    preds = clf.predict(test)
    # print(preds)

    fis = clf.get_best_fuzzy_system()

    print(fis)

    FISViewer(fis).show()

    # Evaluate accuracy
    print(accuracy_score(test_labels, preds))


def gs():
    # Load dataset
    data = load_breast_cancer()
    # data = load_iris()

    # Organize our data
    label_names = data['target_names']
    labels = data['target']
    feature_names = data['feature_names']
    features = data['data']

    # Split our data
    train, test, train_labels, test_labels = train_test_split(features,
                                                              labels,
                                                              test_size=0.33)

    tuned_params = {"dont_care_prob": [None, 0.1, 0.5, 0.8, 0.99]}

    gs = GridSearchCV(FugeClassifier(n_rules=3, n_generations=200, pop_size=100,
                                     n_labels_per_mf=3, verbose=True),
                      tuned_params)

    # for some reason I have to pass y with same shape
    # otherwise gridsearch throws an error. Not sure why.
    gs.fit(train, y=train_labels)

    print(gs.best_params_)  # {'intValue': -10} # and that is what we expect :)


if __name__ == '__main__':
    main()
    # gs()
