import numpy as np

from evo.helpers.ifs_utils import IFSUtils


def run_without_evo():
    # inputs: "SL", "SW", "PL", "PW"
    # outputs: setosa, versicolor, virginica

    labels = ["low", "medium", "high", "dc"]

    # ds_train, ds_test = Iris2PFDataset()
    # observations = ds_train.X

    import pandas as pd

    fname = r"/home/gary/CI4CB/PyFUGE/fuzzy_systems/examples/iris/iris.data"
    iris_dataset = pd.read_csv(fname, sep=",",
                               names=["SL", "SW", "PL", "PW", "OUT"])

    X_names = ["SL", "SW", "PL", "PW"]
    X = iris_dataset[X_names].values

    observations = X
    # print("obs", observations)

    iris_vars_range = np.empty((observations.shape[1], 2))
    iris_vars_range[:, 0] = observations.ptp(axis=0)
    iris_vars_range[:, 1] = observations.min(axis=0)

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
        dc_idx=-1
    )

    print((time() - t0) * 1000, "ms")

    # print(predicted_outputs[0])

    # np.savetxt("/tmp/pyfuge.csv", predicted_outputs, delimiter=",")


if __name__ == '__main__':
    run_without_evo()
