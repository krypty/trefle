import pyfuge_c

import numpy as np

np.set_printoptions(precision=2, suppress=True)


def predict_native(ind, observations, n_rules, max_vars_per_rule, n_labels,
                   n_consequents, default_rule_cons, vars_ranges,
                   labels_weights):
    # we have decided the don't care index will always be the last label
    dc_idx = n_labels - 1

    y_preds = pyfuge_c.bind_predict(
        np.array(ind, dtype=np.float32), observations,
        n_rules, max_vars_per_rule, n_labels, n_consequents,
        default_rule_cons, vars_ranges, labels_weights,
        dc_idx
    )

    return y_preds
