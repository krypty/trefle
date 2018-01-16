from evo.IFS2 import predict, IFSUtils


# class IFS:
#     def __init__(self):
#         self._rules = []
#         self._def_rule = None
#         self._consequents = []
#
#     def predict(self, dataset: PFDataset):
#         pass


# class FuzzyRule:
#     def __init__(self):
#         """
#
#         """
#         self.antecedents = []
#         self.consequent = None


# class FuzzyVariable:
#     def __init__(self, mf_params: np.ndarray):
#         """
#         # TODO: make sure that mf_params is sorted or sort it !
#         :param mf_params: np.ndarray (Nx1) where N is the number of fuzzy values
#         (e.g. 4 for very low, low, medium and high). mf_params contains the
#         p parameters i.e. the points/boundaries where a fuzzy value changes (e.g
#         low to medium). mf_params values must be scaled to the variables range.
#         Before, you might want to do sth like: x = x * v.ptp() + v.min()
#
#           ^
#           | low      medium           high
#         1 |XXXXX       X          XXXXXXXXXXXX
#           |     X     X  X       XX
#           |      X   X    X    XX
#           |       X X      XX X
#           |       XX        XXX
#           |      X  X     XX   XX
#           |     X    X XX       XX
#           |    X       X          XX
#         0 +-------------------------------------->
#                p1     p2          p3
#         """
#         self._mf_params = mf_params
#
#     @staticmethod
#     def in_values(size):
#         """
#         Return a list of in_values for a given size. For example, size=3
#         -> low = [1, 0, 0], med = [0, 1, 0] and high = [0, 0, 1]
#         This function is cached because it will be called a lot of time
#         and the results are always the same.
#         :param size:
#         :return:
#         """
#         return np.eye(size)


if __name__ == '__main__':
    import numpy as np

    np.set_printoptions(precision=2)

    ## CASES
    observations = np.array([
        [5, 275],
        [35, 200],
        [43, 222],
        [15, 123],
    ])  # NxM: N=#cases, M=#vars, probably X_train

    # FROM USER
    individual = np.random.rand(4 * 2 + 3 * 2 + 1 * 3)  # TODO

    print("ind [{}] {}".format(len(individual), individual))
    MAX_VARS_PER_RULE = 2
    N_RULES = 3

    labels_names = ["L", "M", "H", "VH", "DC"]
    N_LABELS = len(labels_names)
    labels_weights = np.ones(N_LABELS)
    N_CONSEQUENTS = 1

    DC_IDX = N_LABELS - 1  # FIXME: current assumption says that last idx is
    # DC, but we should let the user decide. later...
    # print("DC_IDX", DC_IDX)

    vars_range_func = IFSUtils.create_vars_range_getter(dataset=observations)

    from time import sleep, time

    t0 = time()
    predict(ind=individual,
            dataset=observations,
            n_rules=N_RULES,
            max_vars_per_rule=MAX_VARS_PER_RULE,
            n_labels=N_LABELS,
            n_consequents=N_CONSEQUENTS,
            vars_range_getter=vars_range_func,
            default_rule_cons=0,
            labels_weights=labels_weights)

    tick = (time() - t0) * 1000
    print("{} ms".format(tick))

    sleep(0.2)
    assert False

    # ifs = IFS()

    evo_ants = np.array([
        [0.1, 0.2],  # r0
        [1.0, 0.3],  # r1
        [0.4, 0.5],  # r2
    ])
    # MAX_VARS_PER_RULE = evo_ants.shape[1]

    ifs_ants_idx = IFSUtils.evo_ants2ifs_ants(evo_ants, labels_weights)

    ifs_ants_idx = np.array([
        [0, DC_IDX],  # r0: mf_idx_v0, mf_idx_v1
        [4, 1],  # r1: mf_idx_v0, mf_idx_v1
        [2, 0],  # r2: mf_idx_v0, mf_idx_v1
        # [DC_IDX, DC_IDX]# --> TODO: must ignore this kind of rule
    ])
    # print(ifs_ants_idx)
    # print(ifs_ants_idx.shape)

    evo_mfs = np.array([
        [0.1, 0.2, 0.4, 0.4],
        [0.22, 0.33, 0.44, 0.55],
    ])

    #  evo_mfs2ifs_mfs(evo_mfs, vars_range_getter)

    # vars_range_func should be declared outside the evolution process
    vars_range_func = IFSUtils.create_vars_range_getter(dataset=observations)
    in_values = IFSUtils.evo_mfs2ifs_mfs(evo_mfs, vars_range_func)

    print("in_values")
    print(in_values)
    # in_values = np.array([
    #     [10, 20, 33, 40],
    #     [200, 300, 400, 403],
    # ])

    assert in_values.shape[1] == len(labels_names) - 1  # drop DC label

    # test_evo_ants2ifs_ants()
    # test_evo_ants2ifs_ants_2()

    for obs_i in observations:
        # TODO: extract constants variables/arrays

        # raw_inp = np.array([5, 275])  # v0, v1
        # rule0_ant_idx = ifs_ants_idx[0]
        # print(rule0_ant_idx)  # e.g. [0,1] ->
        mf_values_eye = np.eye(N_LABELS)

        # set DC row to 1. This will neutralize the effect of AND_min
        mf_values_eye[DC_IDX] = 1
        mf_values_eye = mf_values_eye[:, :-1]  # shape = (N_LABELS, N_LABELS-1)

        # print("mf_eye")
        # print(mf_values_eye)
        # print("--")

        # in_values_r0 = np.take(mf_values_eye, rule0_ant_idx, axis=0)
        # print("yolo", in_values_r0)

        fuz_ants = np.zeros_like(ifs_ants_idx).astype(np.float)
        for ri in range(len(ifs_ants_idx)):
            # ignore rule with all antecedents set to DC
            if np.all(ifs_ants_idx[ri] == DC_IDX):
                print("rule {} ignored".format(ri))
                continue

            for vi in range(MAX_VARS_PER_RULE):
                # print("x", in_values[vi])
                # _mf_values = mf_values_eye[rule0_ant_idx[i]]
                _mf_values = mf_values_eye[ifs_ants_idx[ri][vi]]
                # print("y", _mf_values)
                fuz_ants[ri, vi] = np.interp(obs_i[vi], in_values[vi],
                                             _mf_values)

        # print("fuz_ants")
        # print(fuz_ants)

        # fuz_ants
        # [[ 1.    1.  ]
        #  [ 1.    0.75]
        #  [ 0.    0.25]]

        ## RULES ACTIVATION
        rules_act = np.min(fuz_ants, axis=1)
        # print("rules_act")
        # print(rules_act)

        # rule_act
        # [ 1.    0.75  0.  ]

        ## DEFAULT RULE ACTIVATION
        def_rule_act = 1.0 - np.max(rules_act)
        rules_act = np.append(rules_act, def_rule_act)

        ## IMPLICATION
        evo_cons = np.array([
            [0.69],
            [0.111],
            [0.88]
        ])
        ifs_cons = IFSUtils.evo_cons2ifs_cons(evo_cons)
        # print(ifs_cons)

        MALIGN = 0
        BENIGN = 1
        ifs_cons = np.array([
            [MALIGN],
            [BENIGN],
            [MALIGN],
        ])  # NxM: N=#rules (+1 for def_rule), M=1 ~~, only 1 consequent is supported *for now*~~

        assert evo_cons.shape == ifs_cons.shape
        assert evo_cons.shape[0] == ifs_ants_idx.shape[0], \
            "#rules != #consequents"

        # TODO: get default_rule_consequent from user
        def_rule_cons = MALIGN
        ifs_cons = np.append(ifs_cons, def_rule_cons)
        ## AGGREGATION
        # mu_rules =

        ## AGGREGATION and DEFUZZIFICATION
        defuzz_out = rules_act.dot(ifs_cons) / np.sum(rules_act)
        # print("defuzz_out")
        print("obs {}: {}".format(obs_i, defuzz_out))
