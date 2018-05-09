import numpy as np

from pyfuge.evo.helpers import NativeIFSUtils
from pyfuge.evo.helpers.ifs_utils import IFSUtils
from pyfuge.fuzzy_systems.view.fis_viewer import FISViewer


def run_without_evo():
    # inputs: "SL", "SW", "PL", "PW"
    # outputs: setosa, versicolor, virginica

    labels = ["low", "medium", "high", "dc"]

    # ds_train, ds_test = Iris2PFDataset()
    # observations = ds_train.X

    import pandas as pd

    fname = r"/home/gary/CI4CB/PyFUGE/datasets/iris.data"
    iris_dataset = pd.read_csv(fname, sep=",",
                               names=["SL", "SW", "PL", "PW", "OUT"])

    X_names = ["SL", "SW", "PL", "PW"]
    X = iris_dataset[X_names].values

    observations = X
    # print("obs", observations)

    iris_vars_range = IFSUtils.compute_vars_range(observations)

    ind = []

    # mfs
    ind.extend([0, 0, 0])  # v0 SL
    ind.extend([0, 0, 0])  # v1 SW
    ind.extend([0, 0, 0])  # v2 PL
    ind.extend([0, 0, 0])  # v3 PW

    a = np.array(ind)
    # print(a.reshape(4, -1))

    # ants
    dc_value = 1.0
    low_value = 0.0
    med_value = 0.26
    high_value = 0.51
    ind.extend([dc_value, dc_value, dc_value, low_value])  # r0
    ind.extend([dc_value, dc_value, low_value, med_value])  # r1
    ind.extend([high_value, med_value, low_value, high_value])  # r2

    # cons
    ind.extend([0.9, 0.1, 0.1])  # r0
    ind.extend([0.1, 0.9, 0.1])  # r1
    ind.extend([0.1, 0.9, 0.1])  # r2

    default_rule_cons = np.array([0, 0, 1])

    from time import time

    # print("len ind", len(ind))
    t0 = time()

    n_labels = len(labels)

    predicted_outputs = IFSUtils.predict(
        ind=ind,
        observations=observations,
        n_rules=3,
        max_vars_per_rule=4,
        n_labels=n_labels,
        n_consequents=3,
        default_rule_cons=default_rule_cons,
        vars_ranges=iris_vars_range,
        labels_weights=np.ones(n_labels),
    )

    print((time() - t0) * 1000, "ms")

    print(predicted_outputs[0])

    # np.savetxt("/tmp/pyfuge.csv", predicted_outputs, delimiter=",")


def _compute_accuracy(y_true, y_pred):
    # y_pred_bin = np.where(y_pred >= 0.5, 1, 0)
    #
    # n_good = 0
    # for row in range(y_pred.shape[0]):
    #     if np.all(np.equal(y_pred_bin[row], y_true[row])):
    #         n_good += 1
    # return n_good / float(y_pred.shape[0])

    # PER CLASS ACC
    y_pred_bin = np.where(y_pred >= 0.5, 1, 0)

    # exclusive prediction. highest to 1, others to 0. don't allow e.g. [0 1 1]
    # y_pred_bin = (y_pred / y_pred.max(axis=1).reshape(-1, 1)).astype(int)

    n_classes = y_pred_bin.shape[1]
    acc_per_class = [-1] * n_classes
    for c_idx in range(n_classes):
        acc = (y_true[:, c_idx] == y_pred_bin[:, c_idx]).mean()
        acc_per_class[c_idx] = acc

    return acc_per_class


def run_with_simple_evo():
    from time import time
    from pyfuge.evo.playground.iris_ifs import Iris2PFDataset
    from pyfuge.evo.experiment.pyfuge_simple_ea_ind2ifs import \
        PyFUGESimpleEAInd2IFS
    from pyfuge.evo.experiment.base.simple_experiment import SimpleEAExperiment
    from pyfuge.evo.fitness_evaluator.pyfuge_fitness_evaluator import \
        PyFUGEFitnessEvaluator

    t0 = time()
    tick = lambda: print((time() - t0) * 1000)

    ##
    ## LOAD DATASET
    ##
    ds_train, ds_test = Iris2PFDataset(
        fname=r"../../datasets/iris.data")

    ##
    ## EXPERIMENT PARAMETERS
    ##
    n_vars = ds_train.N_VARS
    n_rules = 3
    n_max_vars_per_rule = 2
    mf_label_names = ["LOW", "HIGH", "DC"]
    default_rule_output = [0, 1, 0]  # [setosa, versicolor, virginica]
    labels_weights = np.ones(len(mf_label_names))

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
        N_POP=400,
        N_GEN=50
    )

    top_n = exp.get_top_n()

    fis_li = []
    for ind in top_n:
        print("ind ({}): {}".format(ind.fitness, ind))
        fis = pyfuge_ind_2_ifs.convert(ind)
        fis.describe()
        FISViewer(fis).show()

        fis_li.append(fis)

    ##
    ## VALIDATION PHASE
    ##

    # make sure the var_range is still set to training set. If not, we cheat
    var_range_train = IFSUtils.compute_vars_range(ds_train.X)

    for ind in top_n:
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

        acc = _compute_accuracy(ds_test.y, y_pred_test)
        print("acc ", acc)

    tick()


if __name__ == '__main__':
    # run_without_evo()
    run_with_simple_evo()
