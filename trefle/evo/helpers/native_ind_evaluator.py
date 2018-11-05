from trefle.fuzzy_engine import FISEvalWrapper


class NativeIndEvaluator:
    def __init__(self, ind_n, observations, n_rules, max_vars_per_rule,
                 n_labels, n_consequents, default_rule_cons, vars_ranges,
                 labels_weights):
        """
        Make the prediction (i.e. given an individual, it creates a FIS and
        finally returns the predictions aka y_preds) using a C++ engine (faster)

        Assumptions:
        - singleton (aggregation and defuzzification done according the book)
        - classifier type (multiple consequents in [0, 1])
        - AND_min is the implication/rule operator
        - mandatory default rule
        - not operator unsupported FIXME ?
        - max_vars_per_rule <= n_rules
        -
        -
        :param ind_n: length of an individual
        :param observations: a NxM np.array N=n_observations, M=n_vars
        :param max_vars_per_rule: maximum of variables per rule. Must be <= n_vars
        :param n_labels: number of linguistic labels (e.g. LOW, MEDIUM, HIGH,
        DONT_CARE). You must include the don't care label e.g.
        ["low", "medium", "high"] --> 3 + 1 (don't care) --> n_labels=4
        :param n_consequents:
        :param default_rule_cons: an np.array defining the consequents for the
        default rule. E.g. [0, 0, 1] if there is 3 consequents.
        Each consequent must be either 0 or 1 since IFS is a classifier
        type singleton fuzzy system
        :param vars_ranges: a Nx2 np.array where each row contains the ith
        variable ptp (range) and minimum. It will be used to scale a
        float in [0, 1]. Example, [[v0_ptp, v0_min], [v1_ptp, v1_min]].
        :param labels_weights: an array of length n_labels. Set the labels
        weights. For example, [1, 1, 4] will set the chance to set an antecedent
        to don't care (DC) label 4 times more often (on average) than the others
        labels. If none is provided, then all labels have the same probability
        to be chosen.

        """
        self._fisevalwrapper = FISEvalWrapper(ind_n, observations, n_rules,
                                              max_vars_per_rule, n_labels,
                                              n_consequents, default_rule_cons,
                                              vars_ranges, labels_weights)

    def predict_native(self, ind):
        """
        Returns the y_preds (predictions) for a given individual
        :param ind: a list of floats with the following format
        ind = [ v0p0, v0p1, v0p2, v1p0, v1p1, v1p2.. a0r0, a1r0,
            a0r1, a1r1,.. c0r0, c1r0, c0r1, c1r1 ]
        len(ind) = ((n_labels-1) * max_vars_per_rule) * n_rules
        + max_vars_per_rule * n_rules + n_consequents * n_rules
        :return: an array of defuzzified outputs (i.e. non-thresholded outputs)
        """

        return self._fisevalwrapper.predict_native(ind)
