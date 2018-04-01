import numpy as np
import pandas as pd

from evo.dataset.pf_dataset import PFDataset
from evo.helpers import pyfuge_ifs_ind2fis
from evo.helpers.ifs_utils import IFSUtils
from fuzzy_systems.view.fis_viewer import FISViewer


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


def CancerDataset(test_size=0.3):
    df = pd.read_csv(r"CancerDiag2_headers.csv", sep=";")

    dfX = df.drop(["out", "CASE_LBL"], axis=1)
    X = dfX.values
    y = df["out"].values.reshape(-1, 1)

    print("y shape", y.shape)
    X_names = dfX.columns.values
    y_names = ["out"]

    # X = preprocessing.scale(X)

    # X_train, X_test, y_train, y_test = train_test_split(X, y,
    #                                                     test_size=test_size)

    X_train, X_test, y_train, y_test = X, X, y, y

    return (PFDataset(X_train, y_train, X_names, y_names),
            PFDataset(X_test, y_test, X_names, y_names))


# @profile(sort="cumulative", filename="/tmp/pyfuge.profile")
def run_with_simple_evo():
    from time import time
    from evo.experiment.pyfuge_simple_ea_ind2ifs import PyFUGESimpleEAInd2IFS
    from evo.experiment.base.simple_experiment import SimpleEAExperiment
    from evo.fitness_evaluator.pyfuge_fitness_evaluator import \
        PyFUGEFitnessEvaluator

    # import random
    # random.seed(10)
    # np.random.seed(10)

    t0 = time()
    tick = lambda: print((time() - t0) * 1000)

    ##
    ## LOAD DATASET
    ##
    ds_train, ds_test = CancerDataset(test_size=0.3)

    ##
    ## EXPERIMENT PARAMETERS
    ##
    n_vars = ds_train.N_VARS
    n_rules = 5
    n_max_vars_per_rule = 2  # FIXME: don't ignore it
    mf_label_names = ["LOW", "HIGH", "DC"]
    default_rule_output = [1]  # [class_0]
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
        N_GEN=10
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
        # fis.describe()
        # FISViewer(fis).show()

        fis_li.append(fis)

    ##
    ## VALIDATION PHASE
    ##

    # make sure the var_range is still set to training set. If not, we cheat
    var_range_train = IFSUtils.compute_vars_range(ds_train.X)

    for ind in top_n[:1]:
        # train
        y_pred_train = IFSUtils.predict(
            ind,
            observations=ds_train.X,
            n_rules=n_rules,
            max_vars_per_rule=n_max_vars_per_rule,
            n_labels=len(mf_label_names),
            n_consequents=len(default_rule_output),
            default_rule_cons=np.array(default_rule_output),
            vars_ranges=var_range_train,
            labels_weights=labels_weights,
            dc_idx=dc_index
        )

        acc = _compute_accuracy(ds_train.y, y_pred_train)
        print("acc train ", acc)

        # test
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
        print("acc test ", acc)


if __name__ == '__main__':
    from time import time

    t0 = time()
    run_with_simple_evo()
    t1 = time() - t0
    print("evo cancer {:.3f} ms".format(t1 * 1000))
