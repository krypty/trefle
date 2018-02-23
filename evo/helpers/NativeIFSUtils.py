import numpy as np

from evo.examples.evo_wine_classifier import WineDataset, _compute_accuracy


def predict_native(ind, observations, n_rules, max_vars_per_rule, n_labels,
                   n_consequents, default_rule_cons, vars_ranges,
                   labels_weights, dc_idx):
    """
    Assumptions:
    - singleton (aggregation and defuzzification done according the book)
    - classifier type (multiple consequents in [0, 1])
    - AND_min is the implication/rule operator
    - mandatory default rule
    - not operator unsupported FIXME ?
    - max_vars_per_rule <= n_rules
    -
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
    :param vars_ranges: a Nx2 np.array where each row contains the ith variable
    ptp (range) and minimum. It will be used to scale a float in [0, 1].
    Example, [[v0_ptp, v0_min], [v1_ptp, v1_min]].
    :param labels_weights: an array of length n_labels. Set the labels weights.
    For example, [1, 1, 4] will set the chance to set an antecedent to
    don't care (DC) label 4 times more often (on average) than the others
    labels. If none is provided, then all labels have the same probability to be
    chosen.
    :param dc_idx: Specify the don't care index in labels_weights array.
    Must be >=0. negative index will not work !
    :return: an array of defuzzified outputs (i.e. non-thresholded outputs)
    """

    return fiseval.predict_native(ind, observations, n_rules, max_vars_per_rule,
                                  n_labels, n_consequents, default_rule_cons,
                                  vars_ranges, labels_weights, dc_idx)


def simple_predict():
    import random
    random.seed(10)
    np.random.seed(10)

    ##
    ## LOAD DATASET
    ##
    ds_train, ds_test = WineDataset(test_size=0.3)

    ##
    ## EXPERIMENT PARAMETERS
    ##
    n_vars = ds_train.N_VARS
    n_rules = 4
    n_max_vars_per_rule = 2  # FIXME: don't ignore it
    mf_label_names = ["LOW", "HIGH", "DC"]
    default_rule_output = [0, 1, 0]  # [class_0, class_1, class_2]
    labels_weights = np.array([1, 1, 6])
    dc_index = len(mf_label_names) - 1

    vars_range = np.empty((ds_train.X.shape[1], 2))
    vars_range[:, 0] = ds_train.X.ptp(axis=0)
    vars_range[:, 1] = ds_train.X.min(axis=0)

    n_labels = len(mf_label_names)

    ind = [0.4014568272507354, 0.02086371261509734, 0.530756810210902,
           0.3093307757344431, 0.08247854740559168, 0.7019633839142663,
           0.926562433327638, 0.15737845617605617, 0.4787422533535671,
           0.0610712568686953, 0.4691174302744965, 0.04202259295859678,
           0.485411080870507, 0.1947562322392462, 0.2796546885297241,
           0.19321534983890798, 0.8936604011385352, 0.5079196233861747,
           0.14807002370423372, 0.49884630907319505, 0.8970395134423204,
           0.14810283014260428, 0.20260527436666653, 0.3993438824173391,
           0.8359652950866331, 0.5277703928781871, 0.16569094659560835,
           0.04154800624837074, 0.07500165639093082, 0.6088183675731145,
           0.20260527436666653, 0.2751761234518699, 0.26322505730724277,
           0.7404770294423513, 0.5423469084949141, 0.7850582352913449,
           0.33320955037131816, 0.800045802482336, 0.20144613206956785,
           0.5118653116546974, 0.9138359393486287, 0.5751809741545238,
           0.39534799576075363, 0.49472705633579606, 0.9345531513137126,
           0.07917707524907425, 0.7888612066117836, 0.7138797819170863,
           0.35577606365064884, 0.4456129199456278, 0.12103564578689219,
           0.2655354079951163, 0.03078118276387254, 0.10843043323231338,
           0.3993438824173391, 0.3499308379303727, 0.4114438718635758,
           0.06286530731930406, 0.7407613066437472, 0.4678698477895328,
           0.012701678061071364, 0.41970911924220033, 0.576052857622314,
           0.16732654271926128, 0.9357197249689737, 0.9280274704509104,
           0.36221890694318704, 0.22951369290821, 0.5079343214900371,
           0.02206681338701333, 0.11537646855553418, 0.207852814358936,
           0.20668270683981682, 0.572672161174031, 0.9018400559075261,
           0.22146349642572494, 0.5120046294389962, 0.7788752662194598,
           0.6864063822585972, 0.44139806968187045, 0.2666588261678885,
           0.08368468785634153, 0.027077398976345424, 0.5470223993105408,
           0.21373491313776627, 0.7295233734199151, 0.3420745660412098,
           0.07917707524907425, 0.14810283014260428, 0.5386148763532943]

    expected_fitness = 0.5698924731182795

    expected_y_pred = [[0., 1., 0.],
                       [0., 0., 1.],
                       [1., 0., 0.],
                       [0., 1., 0.],
                       [1., 0., 0.],
                       [0., 1., 0.],
                       [0., 1., 0.],
                       [0., 1., 0.],
                       [0., 0., 1.],
                       [1., 0., 0.],
                       [0., 1., 0.],
                       [1., 0., 0.],
                       [1., 0., 0.],
                       [0.5, 0., 0.5],
                       [0., 1., 0.],
                       [0., 0., 1.],
                       [0., 0., 1.],
                       [0., 1., 0.],
                       [0., 1., 0.],
                       [0., 0., 1.],
                       [1., 0., 0.],
                       [0., 0., 1.],
                       [1., 0., 0.],
                       [1., 0., 0.],
                       [0., 1., 0.],
                       [0., 0., 1.],
                       [0.5, 0., 0.5],
                       [0., 0., 1.],
                       [0., 0., 1.],
                       [1., 0., 0.],
                       [0., 0., 1.],
                       [0., 0., 1.],
                       [0., 0., 1.],
                       [0., 0., 1.],
                       [0., 1., 0.],
                       [1., 0., 0.],
                       [0., 1., 0.],
                       [0., 0., 1.],
                       [1., 0., 0.],
                       [0., 1., 0.],
                       [0., 1., 0.],
                       [0., 1., 0.],
                       [0., 0., 1.],
                       [0., 1., 0.],
                       [1., 0., 0.],
                       [0., 1., 0.],
                       [0., 0., 1.],
                       [1., 0., 0.],
                       [0., 1., 0.],
                       [0., 1., 0.],
                       [0., 0., 1.],
                       [0., 1., 0.],
                       [0., 0., 1.],
                       [1., 0., 0.]]

    expected_acc_per_class = [0.8333333333333334, 0.7222222222222222,
                              0.8518518518518519]

    predicted_outputs = predict_native(
        ind=ind,
        observations=ds_test.X,
        n_rules=n_rules,
        max_vars_per_rule=n_max_vars_per_rule,
        n_labels=n_labels,
        n_consequents=len(default_rule_output),
        default_rule_cons=np.array(default_rule_output),
        vars_ranges=vars_range,
        labels_weights=labels_weights,
        dc_idx=n_labels - 1
    )

    print(predicted_outputs)

    acc = _compute_accuracy(ds_test.y, predicted_outputs)

    is_close = np.allclose(acc, expected_acc_per_class)
    print("is close", is_close)

    is_close = np.allclose(predicted_outputs, expected_y_pred)
    print("is close", is_close)


if __name__ == '__main__':
    from cpp.FISEval import fiseval

    simple_predict()
