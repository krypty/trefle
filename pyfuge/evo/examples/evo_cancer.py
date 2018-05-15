import os
from pathlib import Path
from time import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pyfuge.evo.dataset.pf_dataset import PFDataset
from pyfuge.evo.helpers.ifs_utils import IFSUtils
from pyfuge.evo.helpers.native_ind_evaluator import NativeIndEvaluator
from pyfuge.fs.view.fis_viewer import FISViewer


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


def load_cancer_dataset(test_size=0.3):
    PRJ_ROOT = Path(__file__).parents[2]
    filename = os.path.join(PRJ_ROOT, "datasets", "CancerDiag2_headers.csv")
    print("dataset file", filename)
    df = pd.read_csv(
        filename,
        sep=";")

    dfX = df.drop(["out", "CASE_LBL"], axis=1)
    X = dfX.values
    y = df["out"].values.reshape(-1, 1)

    print("y shape", y.shape)
    X_names = dfX.columns.values
    y_names = ["out"]

    # X = preprocessing.scale(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size)

    return (PFDataset(X_train, y_train, X_names, y_names),
            PFDataset(X_test, y_test, X_names, y_names))


def run():
    import random

    random.seed(20)
    np.random.seed(20)

    from pyfuge.evo.experiment.simple_fis_individual import \
        SimpleFISIndividual
    from pyfuge.evo.experiment.base.simple_experiment import SimpleEAExperiment
    from pyfuge.evo.fitness_evaluator.pyfuge_fitness_evaluator import \
        PyFUGEFitnessEvaluator

    ##
    ## LOAD DATASET
    ##
    ds_train, ds_test = load_cancer_dataset(test_size=0.3)

    ##
    ## EXPERIMENT PARAMETERS
    ##
    n_vars = ds_train.N_VARS
    n_rules = 4
    n_max_vars_per_rule = 2  # FIXME: don't ignore it
    mf_label_names = ["LOW", "MEDIUM", "HIGH", "DC"]  # DC is always the last
    default_rule_output = [1]  # [class_0]
    labels_weights = np.array([1, 1, 1, 10])

    ##
    ## TRAINING PHASE
    ##
    fis_ind = SimpleFISIndividual(
        n_vars=n_vars,
        n_rules=n_rules,
        n_max_var_per_rule=n_max_vars_per_rule,
        mf_label_names=mf_label_names,
        default_rule_output=default_rule_output,
        dataset=ds_train,
        labels_weights=labels_weights
    )

    t0 = time()
    exp = SimpleEAExperiment(
        dataset=ds_train,
        fis_individual=fis_ind,
        fitevaluator=PyFUGEFitnessEvaluator(),
        N_POP=400,
        N_GEN=100,
        HOF=3
    )
    t1 = time() - t0
    print("evo cancer {:.3f} ms".format(t1 * 1000))

    top_n = exp.get_top_n()

    fis_li = []
    for ind in top_n[:1]:
        print("ind ({}): {}".format(ind.fitness, ind))
        fis = fis_ind.convert_to_fis(ind)
        fis.describe()
        FISViewer(fis).show()

        fis_li.append(fis)

    ##
    ## VALIDATION PHASE
    ##

    # make sure the var_range is still set to training set. If not, we cheat
    var_range_train = IFSUtils.compute_vars_range(ds_train.X)

    ind_evaluator_train = NativeIndEvaluator(
        ind_n=len(ind),
        observations=ds_train.X,
        n_rules=n_rules,
        max_vars_per_rule=n_max_vars_per_rule,
        n_labels=len(mf_label_names),
        n_consequents=len(default_rule_output),
        default_rule_cons=np.array(default_rule_output),
        vars_ranges=var_range_train,
        labels_weights=labels_weights
    )

    ind_evaluator_test = NativeIndEvaluator(
        ind_n=len(ind),
        observations=ds_test.X,
        n_rules=n_rules,
        max_vars_per_rule=n_max_vars_per_rule,
        n_labels=len(mf_label_names),
        n_consequents=len(default_rule_output),
        default_rule_cons=np.array(default_rule_output),
        vars_ranges=var_range_train,
        labels_weights=labels_weights
    )

    for ind in top_n[:1]:
        # train
        y_pred_train = ind_evaluator_train.predict_native(ind)

        acc = _compute_accuracy(ds_train.y, y_pred_train)
        print("acc train ", acc)

        # test

        y_pred_test = ind_evaluator_test.predict_native(ind)

        acc = _compute_accuracy(ds_test.y, y_pred_test)
        print("acc test ", acc)


if __name__ == '__main__':
    t0 = time()
    run()
    t1 = time() - t0
    print("evo cancer {:.3f} ms".format(t1 * 1000))
