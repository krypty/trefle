import pyfuge_c

import numpy as np


class FISEvalWrapper:
    def __init__(self, ind_n, observations, n_rules, max_vars_per_rule,
                 n_labels, n_consequents, default_rule_cons, vars_ranges,
                 labels_weights):
        # we have decided the don't care index will always be the last label
        dc_idx = n_labels - 1

        self._fiseval = pyfuge_c.FISEvalWrapper(
            ind_n, observations, n_rules, max_vars_per_rule, n_labels,
            n_consequents, default_rule_cons, vars_ranges, labels_weights,
            dc_idx
        )

    def predict_native(self, ind):
        y_preds = self._fiseval.bind_predict(np.array(ind, dtype=np.float32))

        return y_preds
