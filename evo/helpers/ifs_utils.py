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

    # @staticmethod
    # def evo_mfs2ifs_mfs(evo_mfs, ifs_ants_idx, vars_range):
    #     """
    #
    #     :param evo_mfs: np.array of floats in [0, 1]
    #     [
    #         [p0r0, p1r0, p2r0],
    #         [p0r1, p1r1, p2r1],
    #     ]
    #     note: p0 ~= "low", p1 ~= "medium",...
    #     :param ifs_ants_idx:
    #     :param vars_range: np.array of floats in [min(var_i), max(var_i)]
    #     [
    #         [p0r0, p1r0, p2r0],
    #         [p0r1, p1r1, p2r1],
    #     ]
    #     note: p0 ~= "low", p1 ~= "medium",...
    #     :return:
    #     """
    #
    #     # print(evo_mfs.shape)
    #     # print(ifs_ants_idx.shape)
    #     # print(vars_range.shape)
    #     #
    #     # print(vars_range)
    #
    #     for i in range(evo_mfs.shape[0] - 1):
    #         # print("lalal", ifs_ants_idx)
    #         selected_var_range = vars_range.take(ifs_ants_idx)
    #         # print("alalalal", selected_var_range)
    #         evo_mfs[:, i] *= selected_var_range[:, 0]
    #         evo_mfs[:, i] += selected_var_range[:, 1]
    #
    #     # sort the in_values to have increasing membership function
    #     evo_mfs.sort(1)
    #     return evo_mfs

    @staticmethod
    def evo_mfs2ifs_mfs_new(evo_mfs, vars_range):
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

        # print(evo_mfs.shape)
        # print(ifs_ants_idx.shape)
        # print(vars_range.shape)
        #
        # print(vars_range)

        # FIXME: this is wrong ! Both the computation and the sort

        # for i in range(evo_mfs.shape[1]):
        #     selected_var_range = vars_range.take(i, axis=0)
        #     evo_mfs[:, i] *= selected_var_range[0]
        #     evo_mfs[:, i] += selected_var_range[1]
        #     evo_mfs[:, i] = np.sort(evo_mfs[:, i])

        # sort the in_values to have increasing membership function
        # evo_mfs.sort(1)

        # FIXED! might be slower
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

    # @staticmethod
    # def extract_ind(ind, n_rules, n_labels, max_vars_per_rule, n_consequents):
    #     # n_labels-1 because we don't generate a MF for DC label
    #     mfs_idx_len = (n_labels - 1) * max_vars_per_rule
    #     evo_mfs = ind[:mfs_idx_len]
    #     # print("lala", len(evo_mfs))
    #     ants_idx_len = n_rules * max_vars_per_rule
    #     ants_idx_end = mfs_idx_len + ants_idx_len
    #     evo_ants = ind[mfs_idx_len:ants_idx_end]
    #     # cons_idx_len = n_rules * n_consequents
    #     evo_cons = ind[ants_idx_end:]
    #     # print("mfs")
    #     evo_mfs = np.array(evo_mfs, dtype=np.float).reshape(max_vars_per_rule,
    #                                                         n_labels - 1)
    #     # print(evo_mfs)
    #     # print("ants")
    #     evo_ants = np.array(evo_ants).reshape(n_rules, max_vars_per_rule)
    #     # print(evo_ants)
    #     # print("cons")
    #     # print(n_rules, n_consequents)
    #     # print(evo_cons)
    #     evo_cons = np.array(evo_cons, dtype=np.float).reshape(n_rules,
    #                                                           n_consequents)
    #     # print(evo_cons)
    #     return evo_ants, evo_cons, evo_mfs

    @staticmethod
    def extract_ind_new(ind, n_vars, n_labels, n_rules, n_consequents):
        # n_labels-1 because we don't generate a MF for DC label
        mfs_idx_len = (n_labels - 1) * n_vars
        evo_mfs = ind[:mfs_idx_len]
        # print("lala", len(evo_mfs))
        ants_idx_len = n_vars * n_rules
        ants_idx_end = mfs_idx_len + ants_idx_len
        evo_ants = ind[mfs_idx_len:ants_idx_end]
        # cons_idx_len = n_rules * n_consequents
        evo_cons = ind[ants_idx_end:]
        # print("mfs")

        evo_mfs = np.array(evo_mfs, dtype=np.float).reshape(n_vars,
                                                            n_labels - 1)
        # print(evo_mfs)
        # print("ants")
        evo_ants = np.array(evo_ants).reshape(n_rules, n_vars)
        # print("evo_ants")
        # print("evo_ants)
        # print("cons")
        # print(n_rules, n_consequents)
        # print(evo_cons)
        evo_cons = np.array(evo_cons, dtype=np.float).reshape(n_rules,
                                                              n_consequents)

        return evo_mfs, evo_ants, evo_cons

    @staticmethod
    def predict(ind, observations, n_rules, max_vars_per_rule, n_labels,
                n_consequents, default_rule_cons, vars_ranges,
                labels_weights,
                dc_idx):
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
        n_obs, n_vars = observations.shape

        # evo_ants, evo_cons, evo_mfs = \
        #     IFSUtils.extract_ind(ind, n_rules, n_labels, max_vars_per_rule,
        #                          n_consequents)

        evo_mfs, evo_ants, evo_cons = IFSUtils.extract_ind_new(ind, n_vars,
                                                               n_labels,
                                                               n_rules,
                                                               n_consequents)

        # CONVERT EVOLUTION ANTS TO IFS ANTS
        ifs_ants_idx = IFSUtils.evo_ants2ifs_ants(evo_ants, labels_weights)

        # CONVERT EVOLUTION MFS TO IFS MFS
        # in_values = IFSUtils.evo_mfs2ifs_mfs(evo_mfs, vars_range_getter)
        # in_values = IFSUtils.evo_mfs2ifs_mfs(evo_mfs, ifs_ants_idx, vars_ranges)

        in_values = IFSUtils.evo_mfs2ifs_mfs_new(evo_mfs, vars_ranges)

        # print("in shape", in_values.shape)
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
        mf_values_eye = np.eye(n_labels)

        # set DC row to 1. This will neutralize the effect of AND_min
        mf_values_eye[dc_idx] = 1

        # TODO make sure -1 is correct index (dc_idx ?)
        mf_values_eye = mf_values_eye[:, :-1]  # shape = (N_LABELS, N_LABELS-1)

        # TODO test this !
        ifs_cons = IFSUtils.evo_cons2ifs_cons(evo_cons)
        ifs_cons = np.append(ifs_cons, default_rule_cons[np.newaxis, :],
                             axis=0)

        defuzzified_outputs = np.full((n_obs, n_consequents), np.nan)
        for obs_i, obs in enumerate(observations):
            # print("tamre", obs.shape)
            # FUZZIFY INPUTS
            # fuz_ants = np.empty_like(ifs_ants_idx, dtype=np.float)

            rules_act = np.empty(n_rules)

            for ri in range(n_rules):
                # print("yolo\n", ifs_ants_idx)
                ants_ri = ifs_ants_idx[ri]
                # print("ants ri\n", ants_ri)
                # print("dci\n", dc_idx)

                # by default all rules are not triggered
                fuz_ants = np.zeros_like(ants_ri)
                # print("fuz_ants", fuz_ants)

                # ignore rule with all antecedents set to DC
                if np.all(ants_ri == dc_idx):
                    # print("rule {} ignored".format(ri))
                    continue

                # # ignore rule with all antecedents set to DC
                # # ants_ri = ants_ri[ants_ri != dc_idx] #FIXME: keep dc otherwise it will break the shape of ants_ri
                # # print("2ants ri\n", ants_ri)
                #
                # if ants_ri.shape[0] == 0:
                #     print("rule {} ignored".format(ri))
                #     continue

                # ants_ri still contains don't care indices

                # non_dc_ants_idx = ants_ri[ants_ri != dc_idx]
                # print("non dc", non_dc_ants_idx)

                # non_dc_ants_idx = np.argwhere(ants_ri != dc_idx).ravel()

                for vi, lv_value_idx in enumerate(ants_ri):
                    # get the mf_values for the given lv_value_idx (e.g. LOW)
                    _mf_values = mf_values_eye[lv_value_idx]
                    # print(_mf_values)
                    fuz_ants[vi] = np.interp(obs[vi], in_values[vi],
                                             _mf_values)

                # for ai_ri, lv_value_idx in np.ndenumerate(ants_ri):
                #     print("ai_ri", ai_ri)
                #     print("lv_value_idx", lv_value_idx)
                #     _mf_values = mf_values_eye[lv_value_idx]
                #     print("mf valu", _mf_values)
                #
                #     # fuz_ants[ri, vi] = np.interp(obs[vi], in_values[vi], _mf_values) #FIXME; np.interp(obs[vi] won't work if max_var_per_rule != n_rules
                #
                #     print("obs", obs)
                #     obs_x_vi = obs[vi]
                #
                #     print(obs_x_vi)
                #     print(in_values[lv_value_idx])
                #
                #     fuz_ants[lv_value_idx] = np.interp(obs_x_vi,
                #                                        in_values[lv_value_idx],
                #                                        _mf_values)  # FIXME; np.interp(obs[vi] won't work if max_var_per_rule != n_rules

                # print(fuz_ants)
                rules_act[ri] = fuz_ants.min(axis=0)
                # RULE ACTIVATION

            # RULES ACTIVATION
            # rules_act = np.min(fuz_ants, axis=1)

            # DEFAULT RULE ACTIVATION
            def_rule_act = 1.0 - np.max(rules_act)
            rules_act = np.append(rules_act, def_rule_act)

            # # TODO: extract this out of the loop
            # # IMPLICATION
            # ifs_cons = IFSUtils.evo_cons2ifs_cons(evo_cons)
            #
            # # add default rule consequent to consequents
            # # print("lala")
            # # print(ifs_cons.shape)
            # # print("def")
            # # reshape because np.append expects same shape
            # # default_rule_cons = default_rule_cons.reshape(1, -1)
            # # print(default_rule_cons.shape)
            # # TODO: extract this out of the loop
            # ifs_cons = np.append(ifs_cons, default_rule_cons[np.newaxis, :],
            #                      axis=0)

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
