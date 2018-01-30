import numpy as np

from evo.helpers.ifs_utils import IFSUtils
from fuzzy_systems.core.fis.fis import AND_min, MIN
from fuzzy_systems.core.fis.singleton_fis import SingletonFIS
from fuzzy_systems.core.linguistic_variables.linguistic_variable import \
    LinguisticVariable
from fuzzy_systems.core.membership_functions.free_shape_mf import FreeShapeMF
from fuzzy_systems.core.membership_functions.singleton_mf import SingletonMF
from fuzzy_systems.core.rules.default_fuzzy_rule import DefaultFuzzyRule
from fuzzy_systems.core.rules.fuzzy_rule import FuzzyRule
from fuzzy_systems.core.rules.fuzzy_rule_element import Antecedent, Consequent


def convert(n_vars, ind, n_rules, n_labels, n_max_vars_per_rule,
            vars_ranges, labels_weights, dc_index, default_rule_cons,
            pretty_vars_names=None, pretty_outputs_names=None):
    n_consequents = len(default_rule_cons)
    evo_mfs, evo_ants, evo_cons = \
        IFSUtils.extract_ind_new(ind, n_vars, n_labels, n_rules, n_consequents)

    # CONVERT EVOLUTION MFS TO IFS MFS
    ifs_ants_idx = IFSUtils.evo_ants2ifs_ants(evo_ants, labels_weights)
    # in_values = IFSUtils.evo_mfs2ifs_mfs(evo_mfs, ifs_ants_idx, vars_ranges)
    in_values = IFSUtils.evo_mfs2ifs_mfs_new(evo_mfs, vars_ranges)

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

    labels = labels_str_dict[n_labels - 1]

    ifs_cons = IFSUtils.evo_cons2ifs_cons(evo_cons)

    rules = []
    for i in range(len(ifs_ants_idx)):
        r = _create_rule(ifs_ants_idx[i], ifs_cons[i], labels, ling_vars,
                         output_vars, dc_index)
        rules.append(r)

    dr_cons = [Consequent(lv_name=output_vars[i], lv_value=cons) for i, cons in
               enumerate(default_rule_cons)]
    dr = DefaultFuzzyRule(cons=dr_cons, impl_func=MIN)

    # print("evo_mfs")
    # print(evo_mfs)
    #
    # print("evo_ants")
    # print(evo_ants)
    #
    # print("evo_cons")
    # print(evo_cons)
    #
    # print("in_values")
    # print(in_values)
    #
    # print("ifs_ants_idx")
    # print(ifs_ants_idx)

    return SingletonFIS(
        rules=rules,
        default_rule=dr,
    )


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


def _create_rule(ant_idx_ri, cons_idx_ri, labels, ling_vars, output_vars,
                 dc_index):
    ants = [Antecedent(ling_vars[i], labels[ant]) for i, ant in
            enumerate(ant_idx_ri) if ant != dc_index]

    # ants = []
    # for i, ant in enumerate(ant_idx_ri):
    #     if ant != dc_index:
    #         print("ant, dcix", ant, dc_index)
    #         toto = ling_vars[i]
    #         titi = labels[ant]
    #         ants.append(Antecedent(toto, titi))

    cons = [Consequent(lv_name=output_vars[i], lv_value=cons) for i, cons in
            enumerate(cons_idx_ri)]

    return FuzzyRule(ant_act_func=AND_min,
                     ants=ants,
                     cons=cons,
                     impl_func=MIN)
