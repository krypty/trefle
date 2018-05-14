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
        ifs_ants = evo_ants.copy()
        for i in np.ndindex(ifs_ants.shape):
            ifs_ants[i] = IFSUtils.unitfloat2idx(ifs_ants[i], weights)

        return ifs_ants.astype(np.int)

    @staticmethod
    def evo_mfs2ifs_mfs(evo_mfs, vars_range):
        """

        :param evo_mfs: np.array of floats in [0, 1]
        [
            [p0v0, p1v0, p2v0],
            [p0v1, p1v1, p2v1],
        ]
        note: p0 ~= "low", p1 ~= "medium",...
        :param ifs_ants_idx:
        :param vars_range: np.array of floats in [min(var_i), max(var_i)]
        [
            [p0v0, p1v0, p2v0],
            [p0v1, p1v1, p2v1],
        ]
        note: p0 ~= "low", p1 ~= "medium",...
        :return:
        """

        for i in range(evo_mfs.shape[0]):
            row = evo_mfs[i]
            for j in range(evo_mfs.shape[1]):
                row[j] = evo_mfs[i, j] * vars_range[i, 0] + vars_range[i, 1]

            row = np.sort(row)
            evo_mfs[i] = row

        return evo_mfs

    @staticmethod
    def evo_cons2ifs_cons(evo_cons):
        """
        Binarize consequents.
        Assumption: IFS is a Fuzzy Classifier (each consequent is [0, 1]
        :param evo_cons:
        :return:
        """
        return np.where(evo_cons >= 0.5, 1.0, 0.0)

    @staticmethod
    def compute_vars_range(observations):
        _vars_range = np.empty((observations.shape[1], 2))
        _vars_range[:, 0] = observations.ptp(axis=0)
        _vars_range[:, 1] = observations.min(axis=0)
        return _vars_range

    @staticmethod
    def extract_ind(ind, n_vars, n_labels, n_rules, n_consequents):
        # n_labels-1 because we don't generate a MF for DC label
        n_true_labels = n_labels - 1

        mfs_idx_len = n_vars * n_true_labels

        evo_mfs = ind[:mfs_idx_len]
        evo_mfs = np.array(evo_mfs, dtype=np.float).reshape(n_vars,
                                                            n_true_labels)

        ants_idx_len = n_vars * n_rules
        ants_idx_end = mfs_idx_len + ants_idx_len

        evo_ants = ind[mfs_idx_len:ants_idx_end]
        evo_ants = np.array(evo_ants).reshape(n_rules, n_vars)

        evo_cons = ind[ants_idx_end:]
        evo_cons = np.array(evo_cons, dtype=np.float).reshape(n_rules,
                                                              n_consequents)

        return evo_mfs, evo_ants, evo_cons

    @staticmethod
    def predict(ind, observations, n_rules, max_vars_per_rule, n_labels,
                n_consequents, default_rule_cons, vars_ranges,
                labels_weights):
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
        # we have decided that the don't care index will always be the last label
        dc_idx = len(n_labels) - 1
        n_obs, n_vars = observations.shape

        evo_mfs, evo_ants, evo_cons = IFSUtils.extract_ind(ind, n_vars,
                                                           n_labels,
                                                           n_rules,
                                                           n_consequents)

        # CONVERT EVOLUTION ANTS TO IFS ANTS
        ifs_ants_idx = IFSUtils.evo_ants2ifs_ants(evo_ants, labels_weights)

        # CONVERT EVOLUTION MFS TO IFS MFS
        in_values = IFSUtils.evo_mfs2ifs_mfs(evo_mfs, vars_ranges)

        # TODO remove me
        assert in_values.shape[1] == n_labels - 1  # drop DC label

        # PREDICT FOR EACH OBSERVATION IN DATASET
        mf_values_eye = np.eye(n_labels)

        # set DC row to 1. This will neutralize the effect of AND_min
        mf_values_eye[dc_idx] = 1

        # TODO make sure -1 is correct index (dc_idx ?)
        mf_values_eye = mf_values_eye[:, :-1]  # shape = (N_LABELS, N_LABELS-1)

        ifs_cons = IFSUtils.evo_cons2ifs_cons(evo_cons)
        ifs_cons = np.append(ifs_cons, default_rule_cons[np.newaxis, :],
                             axis=0)

        defuzzified_outputs = np.full((n_obs, n_consequents), np.nan)
        for obs_i, obs in enumerate(observations):
            # FUZZIFY INPUTS
            rules_act = np.zeros(n_rules, dtype=np.float64)

            for ri in range(n_rules):
                ants_ri = ifs_ants_idx[ri]

                # by default all rules are not triggered
                fuz_ants = np.zeros_like(ants_ri, dtype=np.float64)
                # print("fuz_ants", fuz_ants)

                # ignore rule with all antecedents set to DC
                if np.all(ants_ri == dc_idx):
                    # print("rule {} ignored".format(ri))
                    continue

                for vi, lv_value_idx in enumerate(ants_ri):
                    # get the mf_values for the given lv_value_idx (e.g. LOW)
                    _mf_values = mf_values_eye[lv_value_idx]
                    fuz_ants[vi] = np.interp(obs[vi], in_values[vi],
                                             _mf_values)

                # RULE ACTIVATION
                rules_act[ri] = fuz_ants.min(axis=0)

            # DEFAULT RULE ACTIVATION
            def_rule_act = 1.0 - np.max(rules_act)
            rules_act = np.append(rules_act, def_rule_act)

            # AGGREGATION and DEFUZZIFICATION
            defuzz_out = rules_act.dot(ifs_cons) / np.sum(rules_act)
            defuzzified_outputs[obs_i] = defuzz_out

        return defuzzified_outputs
