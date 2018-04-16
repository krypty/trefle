import numpy as np
import pandas as pd

from pyfuge.evo.dataset.pf_dataset import PFDataset
from pyfuge.evo.helpers import pyfuge_ifs_ind2fis
from pyfuge.evo.helpers.ifs_utils import IFSUtils
from pyfuge.fuzzy_systems.view.fis_viewer import FISViewer


def _compute_accuracy(y_true, y_pred):
    # ACC
    # y_pred_bin = np.where(y_pred >= 0.5, 1, 0)
    #
    # n_good = 0
    # for row in range(y_pred.shape[0]):
    #     if np.all(np.equal(y_pred_bin[row], y_true[row])):
    #         n_good += 1
    # return n_good / float(y_pred.shape[0])

    # PER CLASS ACC
    y_pred_bin = np.where(y_pred >= 0.5, 1, 0)

    n_classes = y_pred_bin.shape[1]
    acc_per_class = [-1] * n_classes
    for c_idx in range(n_classes):
        acc = (y_true[:, c_idx] == y_pred_bin[:, c_idx]).mean()
        acc_per_class[c_idx] = acc

    return acc_per_class


def load_fake_dataset(test_size=0.3):
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    X, y = make_classification(n_samples=600, n_features=1000, n_informative=3,
                               n_redundant=0, n_classes=2,
                               weights=[0.5, 0.5]
                               )
    y = y.reshape(-1, 1)
    print("X shape", X.shape)
    print("y shape", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size)

    return (PFDataset(X_train, y_train),
            PFDataset(X_test, y_test))


def load_wine_dataset(test_size=0.3):
    from sklearn.datasets import load_wine as load_ds
    from sklearn.model_selection import train_test_split

    dataset = load_ds()

    X = dataset.data
    y = pd.get_dummies(dataset.target).values

    print("X shape", X.shape)
    print("y shape", y.shape)

    X_names = dataset.feature_names
    y_names = dataset.target_names

    # X = preprocessing.scale(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size)

    return (PFDataset(X_train, y_train, X_names, y_names),
            PFDataset(X_test, y_test, X_names, y_names))


# @profile(sort="cumulative", filename="/tmp/pyfuge.profile")
def run_with_simple_evo():
    from time import time
    from pyfuge.evo.experiment.pyfuge_simple_ea_ind2ifs import \
        PyFUGESimpleEAInd2IFS
    from pyfuge.evo.experiment.base.simple_experiment import SimpleEAExperiment
    from pyfuge.evo.fitness_evaluator.pyfuge_fitness_evaluator import \
        PyFUGEFitnessEvaluator

    import random
    random.seed(10)
    np.random.seed(10)

    t0 = time()
    tick = lambda: print((time() - t0) * 1000)

    ##
    ## LOAD DATASET
    ##
    # ds_train, ds_test = loadWineDataset(test_size=0.3)
    ds_train, ds_test = load_fake_dataset(test_size=0.3)

    ##
    ## EXPERIMENT PARAMETERS
    ##
    n_vars = ds_train.N_VARS
    n_rules = 4
    n_max_vars_per_rule = 2  # FIXME: don't ignore it
    mf_label_names = ["LOW", "HIGH", "DC"]
    default_rule_output = [1]  # [class_0, class_1, class_2]
    labels_weights = np.array([1, 1, 10])
    dc_index = len(mf_label_names) - 1

    ##
    ## TRAINING PHASE
    ##
    pyfuge_ind_2_ifs = PyFUGESimpleEAInd2IFS(
        n_vars=n_vars,
        n_rules=n_rules,
        n_max_var_per_rule=n_max_vars_per_rule,
        mf_label_names=mf_label_names,
        default_rule_output=default_rule_output,
        dataset=ds_train,
        labels_weights=labels_weights
    )

    exp = SimpleEAExperiment(
        dataset=ds_train,
        ind2ifs=pyfuge_ind_2_ifs,
        fitevaluator=PyFUGEFitnessEvaluator(),
        N_POP=200,
        N_GEN=100
    )

    tick()
    top_n = exp.get_top_n()

    fis_li = []
    for ind in top_n[:1]:
        print("ind ({}): {}".format(ind.fitness, ind))
        fis = pyfuge_ifs_ind2fis.convert(
            n_vars=n_vars,
            ind=ind, n_rules=n_rules, n_labels=len(mf_label_names),
            n_max_vars_per_rule=n_max_vars_per_rule,
            vars_ranges=IFSUtils.compute_vars_range(ds_train.X),
            labels_weights=labels_weights,
            dc_index=dc_index, default_rule_cons=default_rule_output,
            pretty_vars_names=ds_train.X_names,
            pretty_outputs_names=ds_train.y_names
        )
        fis.describe()
        FISViewer(fis).show()

        fis_li.append(fis)

    ##
    ## VALIDATION PHASE
    ##

    # make sure the var_range is still set to training set. If not, we cheat
    var_range_train = IFSUtils.compute_vars_range(ds_train.X)

    for ind in top_n[:1]:
        y_pred_test = IFSUtils.predict(
            ind,
            observations=ds_test.X,
            n_rules=n_rules,
            max_vars_per_rule=n_max_vars_per_rule,
            n_labels=len(mf_label_names),
            n_consequents=len(default_rule_output),
            default_rule_cons=np.array(default_rule_output),
            vars_ranges=var_range_train,
            labels_weights=labels_weights,
            dc_idx=dc_index
        )

        print(y_pred_test)

        acc = _compute_accuracy(ds_test.y, y_pred_test)
        print("acc test", acc)


if __name__ == '__main__':
    run_with_simple_evo()
