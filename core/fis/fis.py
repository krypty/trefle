from abc import ABCMeta
from collections import defaultdict
from typing import List

import numpy as np

from core.membership_functions.free_shape_mf import FreeShapeMF
from core.rules.fuzzy_rule import FuzzyRule

COA_func = (lambda v, m: np.sum(np.multiply(v, m)) / np.sum(m), "COA_func")
OR_max = (np.max, "OR_max")
AND_min = (np.min, "AND_min")
MIN = (np.min, "MIN")

ERR_MSG_MUST_PREDICT = "you must use predict() at least once"


class FIS(metaclass=ABCMeta):
    def __init__(self, aggr_func, defuzz_func, rules: List[FuzzyRule]):
        self.__aggr_func = aggr_func
        self.__defuzz_func = defuzz_func
        self.__rules = rules

    @property
    def rules(self):
        return self.__rules

    @property
    def last_crisp_values(self):
        assert self._last_crisp_values is not None, ERR_MSG_MUST_PREDICT
        return self._last_crisp_values

    @property
    def last_implicated_consequents(self):
        assert self._implicated_consequents is not None, ERR_MSG_MUST_PREDICT
        return self._implicated_consequents

    @property
    def last_aggregated_consequents(self):
        assert self._aggregated_consequents is not None, ERR_MSG_MUST_PREDICT
        return self._aggregated_consequents

    @property
    def last_defuzzified_outputs(self):
        assert self._defuzzified_outputs is not None, ERR_MSG_MUST_PREDICT
        return self._defuzzified_outputs

    def predict(self, crisp_values):
        self._last_crisp_values = crisp_values
        # TODO: make FIS multiple output variable compatible

        """

        :param crisp_values: a dict where keys are variables name and values are
        variables values. Example: {"temperature": 19, "sunshine": 60}
        :return:
        """

        rules_implicated_cons = defaultdict(list)

        # FUZZIFY AND ACTIVATE INPUTS THEN IMPLICATE CONSEQUENTS FOR EACH RULE
        for r in self.__rules:
            fuzzified_inputs = r.fuzzify(crisp_values)
            antecedents_activation = r.activate(fuzzified_inputs)
            implicated_consequents = r.implicate(antecedents_activation)
            # print(r)
            # print("impl cons", implicated_consequents)

            for k, v in implicated_consequents.items():
                rules_implicated_cons[k].extend(v)
                # rules_implicated_cons.append(implicated_consequents)

                # grouped by rules
        self._implicated_consequents = rules_implicated_cons

        # AGGREGATE CONSEQUENTS
        aggregated_consequents = {}
        for out_v_name, out_v_mf in rules_implicated_cons.items():
            aggregated_consequents[out_v_name] = self.__aggregate(*out_v_mf)

        self._aggregated_consequents = aggregated_consequents

        # DEFUZZIFY
        defuzzified_outputs = {}
        for out_v_name, out_v_mf in aggregated_consequents.items():
            defuzzified_outputs[out_v_name] = self.__defuzzify(out_v_mf)

        self._defuzzified_outputs = defuzzified_outputs
        return defuzzified_outputs

    def __aggregate(self, *out_var_mf):
        all_in_values = np.concatenate([mf.in_values for mf in out_var_mf])
        min_in, max_in = np.min(all_in_values), np.max(all_in_values)

        aggregated_mf_values = []

        aggregated_in_values = np.linspace(min_in, max_in, 50)
        for x in aggregated_in_values:
            fuzzified_x = [mf.fuzzify(x) for mf in out_var_mf]
            mf = self.__aggr_func(fuzzified_x)
            aggregated_mf_values.append(mf)

        return FreeShapeMF(in_values=aggregated_in_values,
                           mf_values=aggregated_mf_values)

    def __defuzzify(self, aggr_mf):
        # print("in v", aggr_mf.in_values, "mf v", aggr_mf.mf_values)
        return self.__defuzz_func[0](aggr_mf.in_values, aggr_mf.mf_values)
