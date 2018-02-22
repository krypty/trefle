# cython: boundscheck=False, wraparound=False, nonecheck=False, cdivision=True

import numpy as np
cimport numpy as np

from cython.parallel import prange, parallel

ctypedef np.float64_t DTYPE_F_T
ctypedef np.int_t DTYPE_I_T

cdef int _unitfloat2idx(float flt, np.ndarray weights):
    len_weight = len(weights)
    cdef np.ndarray[DTYPE_I_T] weights_norm = (weights / weights.min()).astype(np.int)
    cdef np.ndarray[DTYPE_I_T] indices = np.repeat(np.arange(len_weight), weights_norm)

    cdef float idx = flt * len(indices)
    cdef float safe_idx = max(0, min(len(indices) - 1, idx))

    return indices[int(safe_idx)]

cdef np.ndarray _evo_ants2ifs_ants(np.ndarray evo_ants, weights):
    # TODO Cython
    ifs_ants = evo_ants.copy()
    for i in np.ndindex(ifs_ants.shape):
        ifs_ants[i] = IFSUtils.unitfloat2idx(ifs_ants[i], weights)

    return ifs_ants.astype(np.int)


cdef np.ndarray _evo_mfs2ifs_mfs_new(np.ndarray evo_mfs, np.ndarray vars_range):
    cdef int n_vars = evo_mfs.shape[1]

    cdef np.ndarray[DTYPE_F_T, ndim=1] selected_var_range = np.array((2, 1), dtype=np.float64)
    for i in range(n_vars):
        selected_var_range = vars_range.take(i, axis=0)
        evo_mfs[:, i] *= selected_var_range[0]
        evo_mfs[:, i] += selected_var_range[1]

    # sort the in_values to have increasing membership function
    evo_mfs.sort(1)

    return evo_mfs


cdef _extract_ind_new(ind, int n_vars, int n_labels, int n_rules, int n_consequents):
    # n_labels-1 because we don't generate a MF for DC label
    cdef unsigned int mfs_idx_len = (n_labels - 1) * n_vars
    evo_mfs = ind[:mfs_idx_len]
    # print("lala", len(evo_mfs))
    cdef unsigned int ants_idx_len = n_vars * n_rules
    cdef unsigned int ants_idx_end = mfs_idx_len + ants_idx_len
    evo_ants = ind[mfs_idx_len:ants_idx_end]
    # cons_idx_len = n_rules * n_consequents
    evo_cons = ind[ants_idx_end:]
    # print("mfs")

    cdef np.ndarray[DTYPE_F_T, ndim=2] _evo_mfs = np.array(evo_mfs, dtype=np.float).reshape(n_vars,
                                                        n_labels - 1)
    cdef np.ndarray[DTYPE_F_T, ndim=2] _evo_ants = np.array(evo_ants).reshape(n_rules, n_vars)

    evo_cons = np.array(evo_cons, dtype=np.float).reshape(n_rules,
                                                          n_consequents)

    return _evo_mfs, _evo_ants, evo_cons


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
        return _unitfloat2idx(flt, weights)

    @staticmethod
    def evo_ants2ifs_ants(evo_ants, weights):
        """

        :param evo_ants: np.array of floats in [0,1]
        [
            [a0r0, a1r0, a2r0],
            [a0r1, a1r1, a2r1],
        ]
        :param weights:
        :return: np.array of int in [0,len(weights)-1]
        [
            [a0r0, a1r0, a2r0],
            [a0r1, a1r1, a2r1],
        ]
        """
        return _evo_ants2ifs_ants(evo_ants, weights)

    @staticmethod
    def evo_mfs2ifs_mfs_new(evo_mfs, vars_range):
        """

        :param evo_mfs: np.array of floats in [0, 1]
        [
            [p0r0, p1r0, p2r0],
            [p0r1, p1r1, p2r1],
        ]
        note: p0 ~= "low", p1 ~= "medium",...
        :param ifs_ants_idx:
        :param vars_range: np.array of floats in [min(var_i), max(var_i)]
        [
            [p0r0, p1r0, p2r0],
            [p0r1, p1r1, p2r1],
        ]
        note: p0 ~= "low", p1 ~= "medium",...
        :return:
        """

        return _evo_mfs2ifs_mfs_new(evo_mfs, vars_range)

    @staticmethod
    def evo_cons2ifs_cons(evo_cons):
        """
        Binarize consequents.
        Assumption: IFS is a Fuzzy Classifier (each consequent is [0, 1]
        :param evo_cons:
        :return:
        """
        # TODO Cython
        return np.where(evo_cons >= 0.5, 1, 0)

    @staticmethod
    def compute_vars_range(observations):
        # TODO Cython
        _vars_range = np.empty((observations.shape[1], 2))
        _vars_range[:, 0] = observations.ptp(axis=0)
        _vars_range[:, 1] = observations.min(axis=0)
        return _vars_range

    @staticmethod
    def extract_ind_new(ind, n_vars, n_labels, n_rules, n_consequents):
        return _extract_ind_new(ind, n_vars, n_labels, n_rules, n_consequents)

    @staticmethod
    def predict(
            ind,
            observations,
            int n_rules,
            int max_vars_per_rule,
            int n_labels,
            int n_consequents,
            np.ndarray default_rule_cons,
            np.ndarray vars_ranges,
            np.ndarray labels_weights,
            int dc_idx):
        # FIXME: use max_vars_per_rule
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
        cdef unsigned int n_obs = observations.shape[0]
        cdef unsigned int n_vars = observations.shape[1]

        ind_parts = IFSUtils.extract_ind_new(ind, n_vars, n_labels, n_rules, n_consequents)
        cdef np.ndarray evo_mfs  = ind_parts[0]
        cdef np.ndarray evo_ants = ind_parts[1]
        cdef np.ndarray evo_cons = ind_parts[2]


        cdef int vi = 0


        # CONVERT EVOLUTION ANTS TO IFS ANTS
        cdef np.ndarray ifs_ants_idx = IFSUtils.evo_ants2ifs_ants(evo_ants, labels_weights)

        # CONVERT EVOLUTION MFS TO IFS MFS
        # in_values = IFSUtils.evo_mfs2ifs_mfs(evo_mfs, vars_range_getter)
        # in_values = IFSUtils.evo_mfs2ifs_mfs(evo_mfs, ifs_ants_idx, vars_ranges)

        cdef np.ndarray[DTYPE_F_T, ndim=2] in_values = IFSUtils.evo_mfs2ifs_mfs_new(evo_mfs, vars_ranges)

        # # TODO remove me
        # in_values = np.array([
        #     [4.65, 4.65, 5.81],  # SL
        #     [2.68, 3.74, 4.61],  # SW
        #     [4.68, 5.26, 6.03],  # PL
        #     [0.39, 1.16, 2.03]  # PW
        # ])

        # TODO remove me
        assert in_values.shape[1] == n_labels - 1  # drop DC label

        # PREDICT FOR EACH OBSERVATION IN DATASET
        cdef double[:, :] mf_values_eye = np.eye(n_labels)

        # set DC row to 1. This will neutralize the effect of AND_min
        mf_values_eye[dc_idx] = 1

        # TODO make sure -1 is correct index (dc_idx ?)
        mf_values_eye = mf_values_eye[:, :-1]  # shape = (N_LABELS, N_LABELS-1)

        # TODO test this !
        cdef np.ndarray ifs_cons = IFSUtils.evo_cons2ifs_cons(evo_cons)
        ifs_cons = np.append(ifs_cons, default_rule_cons[np.newaxis, :],
                             axis=0)

        cdef np.ndarray defuzzified_outputs = np.full((n_obs, n_consequents), np.nan)


        cdef np.ndarray[DTYPE_F_T] obs
        cdef np.ndarray[DTYPE_F_T] rules_act

        cdef np.ndarray[DTYPE_I_T] ants_ri
        cdef int lv_value_idx = -1
        cdef float[:] _mf_values


        cdef double[:] fuz_ants



        for obs_i in range(n_obs):
            obs = observations[obs_i]

            # FUZZIFY INPUTS
            rules_act = np.empty(n_rules, dtype=np.float64)

            for ri in range(n_rules):
                ants_ri = ifs_ants_idx[ri]

                # by default all rules are not triggered
                fuz_ants = np.zeros_like(ants_ri, dtype=np.float)

                # ignore rule with all antecedents set to DC
                if np.all(ants_ri == dc_idx):
                    # print("rule {} ignored".format(ri))
                    continue


                for vi in range(n_vars):
                    lv_value_idx = ants_ri[vi]
                    # get the mf_values for the given lv_value_idx (e.g. LOW)
                    # _mf_values =
                    # print(_mf_values)
                    # fuz_ants[vi] = np.interp(obs[vi], in_values[vi],
                    #                          _mf_values)
                    np.interp(obs[vi], in_values[vi], mf_values_eye[lv_value_idx])


                # print(fuz_ants)
                rules_act[ri] = np.min(fuz_ants, axis=0)
                # RULE ACTIVATION

            # RULES ACTIVATION
            # rules_act = np.min(fuz_ants, axis=1)

            # DEFAULT RULE ACTIVATION
            def_rule_act = 1.0 - np.max(rules_act)
            rules_act = np.append(rules_act, def_rule_act)

            # IMPLICATION (done outside this loop)

            # AGGREGATION and DEFUZZIFICATION
            defuzz_out = np.divide(rules_act.dot(ifs_cons), np.sum(rules_act))
            # print("obs {}: {}".format(obs, defuzz_out))
            defuzzified_outputs[obs_i] = defuzz_out

        # print("defuz outs")
        # print(defuzzified_outputs)
        # print("-" * 20)
        return defuzzified_outputs
