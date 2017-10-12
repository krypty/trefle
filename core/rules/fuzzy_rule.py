from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Callable, Tuple

from core.membership_functions.free_shape_mf import FreeShapeMF
from core.rules.fuzzy_rule_element import Antecedent, Consequent


class FuzzyRule:
    def __init__(self,
                 ants: List[Antecedent],
                 ant_act_func: Tuple[Callable, str],
                 cons: List[Consequent],
                 impl_func: Tuple[Callable, str]):
        """
        Define a fuzzy rule

        Assumptions:
        * the antecedent's activation function is the same for all consequents
        * multiple antecedents and consequents can be used for a single rule

        :param ants: a list of Antecedent
        :param ant_act_func:
        :param cons:
        :param impl_func:
        """
        self._ants = ants
        self._ant_act_func = ant_act_func
        self._cons = cons
        self._impl_func = impl_func

    @property
    def antecedents(self):
        return self._ants

    @property
    def consequents(self):
        return self._cons

    def fuzzify(self, crisp_inputs: Dict[str, float]) -> List[float]:
        """

        :rtype: FuzzyRule
        :param crisp_inputs: the rule's antecedents crisps inputs values i.e. a
        user's/dataset sample input.
        :return: fuzzified inputs for this particular rule
        """

        fuzzified_inputs_for_rule = []
        for a in self.antecedents:
            in_val = crisp_inputs[a.lv_name.name]

            if a.is_not:
                fuzzified_input = 1.0 - a.lv_name[a.lv_value].fuzzify(in_val)
            else:
                fuzzified_input = a.lv_name[a.lv_value].fuzzify(in_val)

            fuzzified_inputs_for_rule.append(fuzzified_input)

        return fuzzified_inputs_for_rule

    def activate(self, fuzzified_inputs):
        """
        Compute and return the antecedents activation for this rule
        :param fuzzified_inputs:
        :return: a scalar that represents the antecedents activation
        """
        # cannot set this value arbitrarily because it can lead to wrong results
        # example: if rule_ant_val is set to 0 and AND operator is used, the
        # result will always be 0
        # ????  known drawback: when i = 0 it's like we do: ant_act_func(ant0, ant0)
        # ????  FIXME: this will not work with probor function -> now should work if we start at range(1, ..)
        ant_val = fuzzified_inputs[0]

        # apply the rule antecedent function using a sliding window of size 2
        for i in range(1, len(fuzzified_inputs)):
            ant_val = self._ant_act_func[0]([ant_val, fuzzified_inputs[i]])

        return ant_val

    def implicate(self, antecedents_activation):
        """
        Compute and return the rule's implication for all the consequents for
        this particular rule.
        A rule's implication is computed as follow:
        RI_for_consequent_C =  implication_func(antecedents_activation, C)

        :param antecedents_activation: the rule's antecedents activation value.
        So the scalar value returned by self.activate()
        :return: a list (in the same order as the consequents were given in
        the constructor) of FreeShapeMF objects that represents the rule's
        consequents (i.e. output variables) after applying the implication
        operation
        """

        implicated_consequents = defaultdict(list)

        # TODO: improve the comment below
        # (conseq, conseq_label) is (linguistic variable, linguistic value used in this conseq)
        for con in self._cons:
            conseq, conseq_label = con.lv_name, con.lv_value
            ling_value = conseq[conseq_label]
            in_values = deepcopy(ling_value.in_values)  # deepcopy needed ?
            mf_values = [self._impl_func[0]([val, antecedents_activation]) for
                         val in ling_value.mf_values]

            implicated_consequents[conseq.name].append(
                FreeShapeMF(in_values, mf_values))

        return implicated_consequents

    def get_output_variable_names(self):
        return [con.lv_value.name for con in self.consequents]

    def __repr__(self):
        text = "IF ({}) \n" \
               "THEN ({})"

        ants_text = " {} ".format(self._ant_act_func[1]).join(
            ["{} is {}".format(a.lv_name.name, a.lv_value) for a in
             self.antecedents])

        cons_text = " {} ".format(self._impl_func[1]).join(
            ["{} is {}".format(c.lv_name.name, c.lv_value) for c in
             self.consequents])

        return text.format(ants_text, cons_text)
