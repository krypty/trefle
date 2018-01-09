import numpy as np

from evo.Evo import PFDataset


class IFSUtils:
    @staticmethod
    def unitfloat2idx(flt, weights):
        """

        :param flt: a float number in [0, 1]
        :param weights: an array of weights e.g. [1 1 0.5] or [2 1 1 4]
        :return:
        """
        len_weight = len(weights)
        weights_norm = (weights / weights.min()).astype(np.int)
        indices = np.repeat(np.arange(len_weight), weights_norm)
        return indices[int(round(flt * (len_weight - 1)))]

    @staticmethod
    def evo_ants2ifs_ants(evo_ants, weights):
        ifs_ants = evo_ants.copy()
        for i in np.ndindex(ifs_ants.shape):
            ifs_ants[i] = IFSUtils.unitfloat2idx(ifs_ants[i], weights)

        return ifs_ants.astype(np.int)


class IFS:
    def __init__(self):
        self._rules = []
        self._def_rule = None
        self._consequents = []

    def predict(self, dataset: PFDataset):
        pass


class FuzzyRule:
    def __init__(self):
        """

        """
        self.antecedents = []
        self.consequent = None


class FuzzyVariable:
    def __init__(self, mf_params: np.ndarray):
        """
        # TODO: make sure that mf_params is sorted or sort it !
        :param mf_params: np.ndarray (Nx1) where N is the number of fuzzy values
        (e.g. 4 for very low, low, medium and high). mf_params contains the
        p parameters i.e. the points/boundaries where a fuzzy value changes (e.g
        low to medium). mf_params values must be scaled to the variables range.
        Before, you might want to do sth like: x = x * v.ptp() + v.min()

          ^
          | low      medium           high
        1 |XXXXX       X          XXXXXXXXXXXX
          |     X     X  X       XX
          |      X   X    X    XX
          |       X X      XX X
          |       XX        XXX
          |      X  X     XX   XX
          |     X    X XX       XX
          |    X       X          XX
        0 +-------------------------------------->
               p1     p2          p3
        """
        self._mf_params = mf_params

    @staticmethod
    def in_values(size):
        """
        Return a list of in_values for a given size. For example, size=3
        -> low = [1, 0, 0], med = [0, 1, 0] and high = [0, 0, 1]
        This function is cached because it will be called a lot of time
        and the results are always the same.
        :param size:
        :return:
        """
        return np.eye(size)


def test_evo_ants2ifs_ants():
    evo_ants = np.array([
        [0.0, 1.0],  # r0
        [0.5, 0.5],  # r1
        [0.25, 0.75],  # r2
    ])

    exp_arr = np.array([
        [0, 2],
        [2, 2],
        [1, 2],
    ])

    labels_names = ["L", "H", "DC"]
    labels_weights = np.ones(len(labels_names))

    ifs_ants = IFSUtils.evo_ants2ifs_ants(evo_ants, labels_weights)

    try:
        assert np.allclose(exp_arr, ifs_ants)
    except AssertionError:
        print("ERROR !")
        print(exp_arr)
        print("^exp---out--v")
        print(ifs_ants)


def test_evo_ants2ifs_ants_2():
    evo_ants = np.array([
        [0.0, 1.0],  # r0
        [0.5, 0.5],  # r1
        [0.25, 0.75],  # r2
    ])

    exp_arr = np.array([
        [0, 3],
        [2, 2],
        [1, 2],
    ])

    labels_weights = np.array([0.5, 0.5, 0.5, 2])
    ifs_ants = IFSUtils.evo_ants2ifs_ants(evo_ants, labels_weights)

    try:
        assert np.allclose(exp_arr, ifs_ants)
    except AssertionError:
        print("ERROR !")
        print(exp_arr)
        print("^exp---out--v")
        print(ifs_ants)


if __name__ == '__main__':
    import numpy as np

    ifs = IFS()

    evo_ants = np.array([
        [0.1, 0.2],  # r0
        [1.0, 0.3],  # r1
        [0.4, 0.5],  # r2
    ])
    MAX_VARS_PER_RULE = evo_ants.shape[1]

    labels_names = ["L", "M", "H", "VH", "DC"]
    N_LABELS = len(labels_names)
    labels_weights = np.ones(N_LABELS)

    ifs_ants_idx = IFSUtils.evo_ants2ifs_ants(evo_ants, labels_weights)

    DC_IDX = N_LABELS - 1  # FIXME: current assumption says that last idx is
    # DC, but we should let the user decide. later...
    # print("DC_IDX", DC_IDX)

    ifs_ants_idx = np.array([
        [0, DC_IDX],  # r0: mf_idx_v0, mf_idx_v1
        [4, 1],  # r1: mf_idx_v0, mf_idx_v1
        [2, 0],  # r2: mf_idx_v0, mf_idx_v1
    ])
    # print(ifs_ants_idx)
    # print(ifs_ants_idx.shape)

    # TODO: implement evo_mfs to ifs_mfs
    # evo_mfs2ifs_mfs(evo_mfs) # evo_mfs is a NxM array N=#vars and M=#labels

    in_values = np.array([
        [10, 20, 33, 40],
        [200, 300, 400, 403],
    ])

    assert in_values.shape[1] == len(labels_names) - 1  # drop DC label

    # test_evo_ants2ifs_ants()
    # test_evo_ants2ifs_ants_2()

    ## CASE 0
    observations = np.array([
        [5, 275],
        [35, 200],
        [43, 222],
        [15, 123],
    ])  # NxM: N=#cases, M=#vars, prolly X_train
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
        # TODO: ifs_cons = evo_cons2ifs_cons(evo_cons)

        MALIGN = 0
        BENIGN = 1
        ifs_cons = np.array([
            [MALIGN],
            [BENIGN],
            [MALIGN],
        ])  # NxM: N=#rules (+1 for def_rule), M=1, only 1 consequent is supported *for now*

        # TODO: get default_rule_consequent from user
        def_rule_cons = MALIGN
        ifs_cons = np.append(ifs_cons, def_rule_cons)
        ## AGGREGATION
        # mu_rules =

        ## AGGREGATION and DEFUZZIFICATION
        defuzz_out = rules_act.dot(ifs_cons) / np.sum(rules_act)
        # print("defuzz_out")
        print("obs {}: {}".format(obs_i, defuzz_out))
