import numpy as np

from pyfuge.evo.helpers import NativeIFSUtils
from pyfuge.evo.helpers.fis_individual import FISIndividual
from pyfuge.evo.helpers.ifs_utils import IFSUtils
from pyfuge.fs.core.fis.fis import MIN, AND_min
from pyfuge.fs.core.fis.singleton_fis import SingletonFIS
from pyfuge.fs.core.lv.linguistic_variable import \
    LinguisticVariable
from pyfuge.fs.core.mf.free_shape_mf import \
    FreeShapeMF
from pyfuge.fs.core.mf.singleton_mf import \
    SingletonMF
from pyfuge.fs.core.rules.default_fuzzy_rule import DefaultFuzzyRule
from pyfuge.fs.core.rules.fuzzy_rule import FuzzyRule
from pyfuge.fs.core.rules.fuzzy_rule_element import Consequent, \
    Antecedent

# this will raise KeyError if invalid
labels_str_dict = {
    2: ("LOW", "HIGH"),
    3: ("LOW", "MEDIUM", "HIGH"),
    4: ("LOW", "MEDIUM", "HIGH", "VERY HIGH"),
    5: ("VERY LOW", "LOW", "MEDIUM", "HIGH", "VERY HIGH"),
}

_consequents = {
    0: SingletonMF(0),
    1: SingletonMF(1)
}


def _create_ling_var(var_name, in_values_vi):
    labels = labels_str_dict[len(in_values_vi)]

    mf_values = np.eye(len(labels))
    ling_values_dict = {}
    for i, labels_i in enumerate(labels):
        ling_values_dict[labels_i] = FreeShapeMF(in_values_vi.tolist(),
                                                 mf_values[i].tolist())

    return LinguisticVariable(var_name, ling_values_dict)


def _create_rule(ant_idx_ri, cons_idx_ri, labels, ling_vars,
                 output_vars,
                 dc_index):
    ants = [Antecedent(ling_vars[i], labels[ant]) for i, ant in
            enumerate(ant_idx_ri) if ant != dc_index]

    cons = [Consequent(lv_name=output_vars[i], lv_value=cons) for i, cons in
            enumerate(cons_idx_ri)]

    return FuzzyRule(ant_act_func=AND_min,
                     ants=ants,
                     cons=cons,
                     impl_func=MIN)


class SimpleFISIndividual(FISIndividual):
    def __init__(self, n_vars, n_rules, n_max_var_per_rule, mf_label_names,
                 default_rule_output, dataset, labels_weights):
        assert n_max_var_per_rule <= n_vars
        assert n_rules >= 1, "you must set at least 1 rule"

        assert len(mf_label_names) == len(labels_weights), \
            "The number of labels must match the number of labels weights"

        if dataset.y.ndim == 1:
            assert False, "y must be a 2D np array. Maybe you should call " \
                          "pd.get_dummies(y)? "

        assert dataset.y.shape[1] == len(default_rule_output), \
            "default rule output must have the same shape as dataset.y"

        self.n_rules = n_rules
        self.n_max_var_per_rule = n_max_var_per_rule
        self.n_labels = len(mf_label_names)
        self.n_consequents = len(default_rule_output)
        self.default_rule = np.array(default_rule_output)
        self.dataset = dataset
        self.labels_weights = labels_weights
        self.n_vars = dataset.N_VARS

        self.vars_range = np.empty((dataset.X.shape[1], 2))
        self.vars_range[:, 0] = self.dataset.X.ptp(axis=0)
        self.vars_range[:, 1] = self.dataset.X.min(axis=0)

        super(SimpleFISIndividual, self).__init__()

        # mf_label_names = ["LOW", "MEDIUM", "HIGH"]
        self._ind_len = 0

        # [pl0, pm0, ph0, pl1, pm1, ph1,..]
        self._ind_len += (self.n_labels - 1) * n_vars

        # [a0r0, a1r0, a2r0, a0r1...]
        self._ind_len += n_vars * n_rules

        self._ind_len += self.n_consequents * n_rules

        # we have decided the don't care index will always be the last label
        self._dc_index = len(mf_label_names) - 1

    def convert_to_fis(self, ind):
        n_consequents = len(self.default_rule)
        evo_mfs, evo_ants, evo_cons = \
            IFSUtils.extract_ind(ind, self.n_vars, self.n_labels,
                                 self.n_rules, self.n_consequents)

        # CONVERT EVOLUTION MFS TO IFS MFS
        ifs_ants_idx = IFSUtils.evo_ants2ifs_ants(evo_ants, self.labels_weights)
        in_values = IFSUtils.evo_mfs2ifs_mfs(evo_mfs, self.vars_range)

        pretty_vars_names = self.dataset.X_names
        pretty_outputs_names = self.dataset.y_names

        def get_var_name(i):
            if pretty_vars_names is None:
                return "v{}".format(i)
            else:
                return pretty_vars_names[i]

        ling_vars = [_create_ling_var(get_var_name(i), in_values_vi) for
                     i, in_values_vi in enumerate(in_values)]

        def get_output_name(i):
            if pretty_vars_names is None:
                return "OUT{}".format(i)
            else:
                return pretty_outputs_names[i]

        output_vars = [LinguisticVariable(name=get_output_name(i),
                                          ling_values_dict=_consequents)
                       for i in range(n_consequents)]

        labels = labels_str_dict[self.n_labels - 1]

        ifs_cons = IFSUtils.evo_cons2ifs_cons(evo_cons)

        rules = []
        for i in range(len(ifs_ants_idx)):
            if (ifs_ants_idx[i] == self._dc_index).all():
                continue
            r = _create_rule(ifs_ants_idx[i], ifs_cons[i], labels,
                             ling_vars, output_vars, self._dc_index)
            rules.append(r)

        dr_cons = [Consequent(lv_name=output_vars[i], lv_value=cons)
                   for i, cons in enumerate(self.default_rule)]
        dr = DefaultFuzzyRule(cons=dr_cons, impl_func=MIN)

        return SingletonFIS(rules=rules, default_rule=dr)

    def predict(self, ind):
        predicted_outputs = NativeIFSUtils.predict_native(
            ind=ind,
            observations=self.dataset.X,
            n_rules=self.n_rules,
            max_vars_per_rule=self.n_max_var_per_rule,
            n_labels=self.n_labels,
            n_consequents=self.n_consequents,
            default_rule_cons=self.default_rule,
            vars_ranges=self.vars_range,
            labels_weights=self.labels_weights,
        )
        return predicted_outputs
