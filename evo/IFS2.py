from functools import lru_cache

import numpy as np


class IFSUtils:
    @staticmethod
    def unitfloat2idx(flt, weights):
        """
        Returns a weighted index between 0 and len(weights)-1. For example,
        flt=0.5 and weights=[1,1,1,1,1] will create a array A=[0 1 2 3 4] and
        will returns 2 because the index at 0.5*len(A) is 2.
        This function also works with unequals weights such as [1,1,1,4].
        The latter will privilege the last index by 4 times.
        Indeed weights will become [0 1 2 3 3 3 3]. This will be particularly
        useful to give DC label a higher probability to be chosen.

        :param flt: a float number in [0, 1]
        :param weights: an array of weights e.g. [1 1 0.5] or [2 1 1 4]
        :return: the weighted index
        """
        len_weight = len(weights)
        weights_norm = (weights / weights.min()).astype(np.int)
        indices = np.repeat(np.arange(len_weight), weights_norm)

        idx = flt * len(indices)
        safe_idx = max(0, min(len(indices) - 1, idx))
        return indices[int(safe_idx)]

    @staticmethod
    def evo_ants2ifs_ants(evo_ants, weights):
        ifs_ants = evo_ants.copy()
        for i in np.ndindex(ifs_ants.shape):
            ifs_ants[i] = IFSUtils.unitfloat2idx(ifs_ants[i], weights)

        return ifs_ants.astype(np.int)

    @staticmethod
    def create_vars_range_getter(dataset):
        @lru_cache(maxsize=None)
        def _vars_getter(vi):
            x = dataset[:, vi]
            return x.ptp(), x.min()

        return _vars_getter

    @staticmethod
    def evo_mfs2ifs_mfs(evo_mfs, vars_range_getter):
        ifs_mfs = np.empty_like(evo_mfs)
        for i in range(ifs_mfs.shape[0]):
            vptp, vmin = vars_range_getter(i)
            ifs_mfs[i] = evo_mfs[i] * vptp + vmin

        return ifs_mfs

    @staticmethod
    def evo_cons2ifs_cons(evo_cons):
        """
        Binarize consequents.
        Assumption: IFS is a Fuzzy Classifier (each consequent is [0, 1]
        :param evo_cons:
        :return:
        """
        return np.where(evo_cons >= 0.5, 1, 0)


def predict(ind, observations, n_rules, max_vars_per_rule, n_labels,
            n_consequents,
            default_rule_cons, vars_range_getter, labels_weights=None,
            dc_idx=-1):
    """
    Assumptions:
    - singleton
    - classifier type (multiple consequents in [0, 1])
    - mandatory default rule
    - not operator unsupported FIXME ?
    - max_vars_per_rule <= n_rules
    -

    :param ind: a list of floats with the following format
    ind = [ v0p0, v0p1, v0p2, v1p0, v1p1, v1p2.. a0r0, a1r0, a0r1, a1r1,.. c0r0,
     c1r0, c0r1, c1r1 ]
    len(ind) = ((n_labels-1) * max_vars_per_rule) * n_rules
    + max_vars_per_rule * n_rules + n_consequents * n_rules

    :param observations: a NxM np.array N=n_observations, M=n_vars
    :param n_rules:
    :param max_vars_per_rule: maximum of variables per rule. Must be <= n_vars
    :param n_labels: number of linguistic labels (e.g. LOW, MEDIUM, HIGH,
    DONT_CARE). You must include the don't care label e.g.
    ["low", "medium", "high"] --> 3 + 1 (don't care) --> n_labels=4
    :param n_consequents:
    :param default_rule_cons: an np.array defining the consequents for the
    default rule. E.g. [0, 0, 1] if there is 3 consequents. Each consequent must
    be either 0 or 1 since IFS is a classifier type singleton
    fuzzy system
    :param vars_range_getter: a function that, given a variable index, returns
    the data range of the variable. You should use the helper function
    IFSUtils.create_vars_range_getter(dataset) to get this done.
    :param labels_weights: an array of length n_labels. Set the labels weights.
    For example, [1, 1, 4] will set the chance to set an antecedent to
    don't care (DC) label 4 times more often (on average) than the others
    labels. If none is provided, then all labels have the same probability to be
    chosen.
    :param dc_idx: Specify the don't care index in labels_weights array
    :return: an array of defuzzified outputs (i.e. non-thresholded outputs)
    """
    n_obs, n_vars = observations.shape

    # TODO: extract it to caller
    if labels_weights is None:
        labels_weights = np.ones(n_labels)

    # n_labels-1 because we don't generate a MF for DC label
    mfs_idx_len = (n_labels - 1) * max_vars_per_rule
    evo_mfs = ind[:mfs_idx_len]
    # print("lala", len(evo_mfs))

    ants_idx_len = n_rules * max_vars_per_rule
    ants_idx_end = mfs_idx_len + ants_idx_len
    evo_ants = ind[mfs_idx_len:ants_idx_end]

    # cons_idx_len = n_rules * n_consequents
    evo_cons = ind[ants_idx_end:]

    # print("mfs")
    evo_mfs = np.array(evo_mfs).reshape(max_vars_per_rule, n_labels - 1)
    # print(evo_mfs)

    # print("ants")
    evo_ants = np.array(evo_ants).reshape(n_rules, max_vars_per_rule)
    # print(evo_ants)

    # print("cons")
    # print(n_rules, n_consequents)
    # print(evo_cons)
    evo_cons = np.array(evo_cons).reshape(n_rules, n_consequents)
    # print(evo_cons)

    # CONVERT EVOLUTION MFS TO IFS MFS
    in_values = IFSUtils.evo_mfs2ifs_mfs(evo_mfs, vars_range_getter)
    # # TODO remove me
    # in_values = np.array([
    #     [4.65, 4.65, 5.81],  # SL
    #     [2.68, 3.74, 4.61],  # SW
    #     [4.68, 5.26, 6.03],  # PL
    #     [0.39, 1.16, 2.03]  # PW
    # ])

    # CONVERT EVOLUTION ANTS TO IFS ANTS
    ifs_ants_idx = IFSUtils.evo_ants2ifs_ants(evo_ants, labels_weights)

    # TODO remove me
    assert in_values.shape[1] == n_labels - 1  # drop DC label

    # PREDICT FOR EACH OBSERVATION IN DATASET
    defuzzified_outputs = np.full((n_obs, n_consequents), np.nan)
    for obs_i, obs in enumerate(observations):
        # FUZZIFY INPUTS
        mf_values_eye = np.eye(n_labels)

        # set DC row to 1. This will neutralize the effect of AND_min
        mf_values_eye[dc_idx] = 1
        mf_values_eye = mf_values_eye[:, :-1]  # shape = (N_LABELS, N_LABELS-1)

        fuz_ants = np.zeros_like(ifs_ants_idx).astype(np.float)
        for ri in range(n_rules):
            # ignore rule with all antecedents set to DC
            if np.all(ifs_ants_idx[ri] == dc_idx):
                print("rule {} ignored".format(ri))
                continue

            for vi in range(max_vars_per_rule):
                _mf_values = mf_values_eye[ifs_ants_idx[ri][vi]]
                fuz_ants[ri, vi] = np.interp(obs[vi], in_values[vi],
                                             _mf_values)

        # RULES ACTIVATION
        rules_act = np.min(fuz_ants, axis=1)

        # DEFAULT RULE ACTIVATION
        def_rule_act = 1.0 - np.max(rules_act)
        rules_act = np.append(rules_act, def_rule_act)

        # IMPLICATION
        ifs_cons = IFSUtils.evo_cons2ifs_cons(evo_cons)

        # TODO: remove me
        assert evo_cons.shape == ifs_cons.shape
        assert evo_cons.shape[0] == ifs_ants_idx.shape[0], \
            "#rules != #consequents"

        # add default rule consequent to consequents
        # print("lala")
        # print(ifs_cons.shape)
        # print("def")
        # reshape because np.append expects same shape
        default_rule_cons = default_rule_cons.reshape(1, -1)
        # print(default_rule_cons.shape)
        ifs_cons = np.append(ifs_cons, default_rule_cons, axis=0)

        # AGGREGATION and DEFUZZIFICATION
        # print("lalala")
        # print(evo_cons)
        # print(rules_act)
        # print("lalala")
        # print(ifs_cons)

        defuzz_out = rules_act.dot(ifs_cons) / np.sum(rules_act)
        # print("obs {}: {}".format(obs, defuzz_out))
        defuzzified_outputs[obs_i] = defuzz_out

    # print("defuz outs")
    # print(defuzzified_outputs)
    # print("-" * 20)
    return defuzzified_outputs


if __name__ == '__main__':
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

    vars_range_getter = IFSUtils.create_vars_range_getter(observations)
    # for i in range(observations.shape[1]):
    #     print(vars_range_getter(i))

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

    print("len ind", len(ind))
    t0 = time()

    predicted_outputs = predict(
        ind=ind,
        observations=observations,
        n_rules=3,
        max_vars_per_rule=4,
        n_labels=len(labels),
        n_consequents=3,
        default_rule_cons=default_rule_cons,
        vars_range_getter=vars_range_getter,
        labels_weights=None,
        dc_idx=-1
    )

    print((time() - t0) * 1000, "ms")

    # print(predicted_outputs[0])

    np.savetxt("/tmp/pyfuge.csv", predicted_outputs, delimiter=",")
