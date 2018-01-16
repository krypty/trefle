import numpy as np

from evo import IFS2, IFS
from evo.IFS2 import IFSUtils
from evo.dataset.pf_dataset import PFDataset
from evo.experiment.simple_experiment import SimpleEAExperiment
from evo.fitness_evaluator.fitness_evaluator import FitnessEvaluator
from evo.helpers.ind_2_ifs import Ind2IFS


def Iris2PFDataset():
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split

    fname = r"../fuzzy_systems/examples/iris/iris.data"
    iris_dataset = pd.read_csv(fname, sep=",",
                               names=["SL", "SW", "PL", "PW", "OUT"])

    X_names = ["SL", "SW", "PL", "PW"]
    X = iris_dataset[X_names].values

    # e.g. "Iris-Setosa" -> "Setosa"
    y = iris_dataset[["OUT"]].apply(axis=1, func=lambda x: x[0][5:]).values

    # print(X.shape)
    # print(y.shape)

    le = LabelEncoder()
    y = le.fit_transform(y)
    y = pd.get_dummies(y).values
    # print(y[0])
    y_names = le.classes_

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0)

    return (PFDataset(X_train, y_train, X_names, y_names),
            PFDataset(X_test, y_test, X_names, y_names))


if __name__ == '__main__':
    class PyFUGEFitnessEvaluator(FitnessEvaluator):

        @staticmethod
        def _compute_metric(y_pred, y_true):
            return -((y_pred - y_true) ** 2).mean(axis=None)

        def eval(self, ifs: IFS, dataset: PFDataset):
            pass

        def eval_fitness(self, y_preds, dataset: PFDataset):
            y_true = dataset.y
            return self._compute_metric(y_preds, y_true)


    class PyFUGESimpleEAInd2IFS(Ind2IFS):
        def __init__(self, n_vars, n_rules, n_max_var_per_rule, mf_label_names,
                     default_rule_output, dataset):
            assert n_max_var_per_rule <= n_vars

            self.n_rules = n_rules
            self.n_max_var_per_rule = n_max_var_per_rule
            self.n_labels = len(mf_label_names)
            self.n_consequents = len(default_rule_output)
            self.default_rule = np.array(default_rule_output)
            self.dataset = dataset
            self._vars_range_getter = IFSUtils.create_vars_range_getter(
                dataset.X)

            super(PyFUGESimpleEAInd2IFS, self).__init__()

            # mf_label_names = ["LOW", "MEDIUM", "HIGH"]
            self._ind_len = 0

            # [pl0, pm0, ph0, pl1, pm1, ph1,..]
            self._ind_len += (self.n_labels - 1) * n_max_var_per_rule

            # [a0r0, a1r0, a2r0, a0r1...]
            self._ind_len += n_max_var_per_rule * n_rules

            self._ind_len += self.n_consequents * n_rules

        def convert(self, ind):
            pass

        def predict(self, ind):
            predicted_outputs = IFS2.predict(
                ind=ind,
                observations=self.dataset.X,
                n_rules=self.n_rules,
                max_vars_per_rule=self.n_max_var_per_rule,
                n_labels=self.n_labels,
                n_consequents=self.n_consequents,
                default_rule_cons=self.default_rule,
                vars_range_getter=self._vars_range_getter,
                labels_weights=None,
                dc_idx=-1
            )

            return predicted_outputs


    ##
    ## TRAINING PHASE
    ##
    ds_train, ds_test = Iris2PFDataset()

    pyfuge_ind_2_ifs = PyFUGESimpleEAInd2IFS(
        n_vars=ds_train.N_VARS,
        n_rules=3,
        n_max_var_per_rule=4,
        mf_label_names=["LOW", "MEDIUM", "HIGH", "DC"],
        default_rule_output=[0, 0, 1],  # setosa, versicolor, virginica
        dataset=ds_train,
    )

    SimpleEAExperiment(
        dataset=ds_train,
        ind2ifs=pyfuge_ind_2_ifs,
        fitevaluator=PyFUGEFitnessEvaluator(),
        N_POP=100,
        N_GEN=10
    )

    ##
    ## VALIDATION PHASE
    ##
