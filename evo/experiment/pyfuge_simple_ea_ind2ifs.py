import numpy as np

from cpp.FISEval import fiseval
from evo.helpers.ifs_utils import IFSUtils
from evo.helpers.ind_2_ifs import Ind2IFS


class PyFUGESimpleEAInd2IFS(Ind2IFS):
    def __init__(self, n_vars, n_rules, n_max_var_per_rule, mf_label_names,
                 default_rule_output, dataset, labels_weights):
        assert n_max_var_per_rule <= n_vars
        assert n_rules >= 1, "you must set at least 1 rule"

        assert len(mf_label_names) == len(labels_weights), \
            "The number of labels must match the number of labels weights"

        assert dataset.y.shape[1] == len(default_rule_output), \
            "default rule output must have the same shape as dataset.y"

        self.n_rules = n_rules
        self.n_max_var_per_rule = n_max_var_per_rule
        self.n_labels = len(mf_label_names)
        self.n_consequents = len(default_rule_output)
        self.default_rule = np.array(default_rule_output)
        self.dataset = dataset
        self.labels_weights = labels_weights

        self.vars_range = np.empty((dataset.X.shape[1], 2))
        self.vars_range[:, 0] = self.dataset.X.ptp(axis=0)
        self.vars_range[:, 1] = self.dataset.X.min(axis=0)

        super(PyFUGESimpleEAInd2IFS, self).__init__()

        # mf_label_names = ["LOW", "MEDIUM", "HIGH"]
        self._ind_len = 0

        # [pl0, pm0, ph0, pl1, pm1, ph1,..]
        self._ind_len += (self.n_labels - 1) * n_vars

        # [a0r0, a1r0, a2r0, a0r1...]
        self._ind_len += n_vars * n_rules

        self._ind_len += self.n_consequents * n_rules

    def convert(self, ind):
        pass

    def predict(self, ind):
        # predicted_outputs = IFSUtils.predict(
        predicted_outputs = fiseval.predict_native(
            ind=ind,
            observations=self.dataset.X,
            n_rules=self.n_rules,
            max_vars_per_rule=self.n_max_var_per_rule,
            n_labels=self.n_labels,
            n_consequents=self.n_consequents,
            default_rule_cons=self.default_rule,
            vars_ranges=self.vars_range,
            labels_weights=self.labels_weights,
            dc_idx=self.n_labels - 1
        )
        return predicted_outputs
