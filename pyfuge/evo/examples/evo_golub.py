import glob

import pandas as pd
from sklearn import preprocessing

from pyfuge.evo.dataset.pf_dataset import PFDataset
from pyfuge.evo.helpers import NativeIFSUtils
from pyfuge.evo.helpers.ifs_utils import IFSUtils


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


def load_golub_dataset():
    def parse_dataset(filenames, le_classes):
        df = pd.read_csv(filenames[0], sep="\t",
                         usecols=["ID_REF", "VALUE"]).transpose().drop("ID_REF",
                                                                       axis=0)
        df.index = [filenames[0].split("/")[-1]]

        for i in range(1, len(filenames)):
            df2 = pd.read_csv(filenames[i], sep="\t",
                              usecols=["ID_REF", "VALUE"]).transpose().drop(
                "ID_REF", axis=0)
            df2.index = [filenames[i].split("/")[-1]]

            df = pd.concat([df, df2])

        X = df.values

        y = [fname[-7:-4] for fname in df.index.values]
        y = le_classes.transform(y)

        return X, y

    le_classes = preprocessing.LabelEncoder()
    le_classes.fit(["AML", "ALL"])

    path_to_golub = r"../../datasets/golub"
    train_filenames = glob.glob(path_to_golub + "/train/*.csv")
    test_filenames = glob.glob(path_to_golub + "/test/*.csv")

    X_train, y_train = parse_dataset(train_filenames, le_classes)
    X_test, y_test = parse_dataset(test_filenames, le_classes)

    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    print(X_train.shape, y_train.shape)

    return (PFDataset(X_train[:, :300], y_train),
            PFDataset(X_test[:, :300], y_test))


def run_with_simple_evo():
    from pyfuge.evo.experiment.pyfuge_simple_ea_ind2ifs import \
        PyFUGESimpleEAInd2IFS
    from pyfuge.evo.experiment.base.simple_experiment import SimpleEAExperiment
    from pyfuge.evo.fitness_evaluator.pyfuge_fitness_evaluator import \
        PyFUGEFitnessEvaluator

    ##
    ## LOAD DATASET
    ##
    ds_train, ds_test = load_golub_dataset()

    ##
    ## EXPERIMENT PARAMETERS
    ##
    n_vars = ds_train.N_VARS
    n_rules = 7
    n_max_vars_per_rule = 2  # FIXME: don't ignore it
    mf_label_names = ["LOW", "HIGH", "DC"]
    default_rule_output = [0]  # [class_0]
    labels_weights = np.array([1, 1, 100])
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
        N_POP=100,
        N_GEN=10,
        HOF=3
    )

    top_n = exp.get_top_n()

    fis_li = []
    for ind in top_n[:1]:
        print("ind ({}): {}".format(ind.fitness, ind))
        fis = pyfuge_ind_2_ifs.convert(ind)
        fis.describe()
        # FISViewer(fis).show()

        fis_li.append(fis)

    ##
    ## VALIDATION PHASE
    ##

    # make sure the var_range is still set to training set. If not, we cheat
    var_range_train = IFSUtils.compute_vars_range(ds_train.X)

    for ind in top_n[:1]:
        # train
        y_pred_train = NativeIFSUtils.predict_native(
            ind,
            observations=ds_train.X,
            n_rules=n_rules,
            max_vars_per_rule=n_max_vars_per_rule,
            n_labels=len(mf_label_names),
            n_consequents=len(default_rule_output),
            default_rule_cons=np.array(default_rule_output),
            vars_ranges=var_range_train,
            labels_weights=labels_weights,
        )

        acc = _compute_accuracy(ds_train.y, y_pred_train)
        print("acc train ", acc)

        # test
        y_pred_test = NativeIFSUtils.predict_native(
            ind,
            observations=ds_test.X,
            n_rules=n_rules,
            max_vars_per_rule=n_max_vars_per_rule,
            n_labels=len(mf_label_names),
            n_consequents=len(default_rule_output),
            default_rule_cons=np.array(default_rule_output),
            vars_ranges=var_range_train,
            labels_weights=labels_weights,
        )

        print(y_pred_test)

        acc = _compute_accuracy(ds_test.y, y_pred_test)
        print("acc test ", acc)

        print("class balance", np.bincount(ds_test.y.ravel()) / len(ds_test.y))


if __name__ == '__main__':
    from time import time
    import numpy as np

    # import random

    # random.seed(20)
    # np.random.seed(20)

    t0 = time()
    run_with_simple_evo()
    t1 = time() - t0
    print("evo golub {:.3f} ms".format(t1 * 1000))
