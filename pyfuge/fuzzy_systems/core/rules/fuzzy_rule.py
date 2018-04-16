from collections import defaultdict
from copy import deepcopy
from typing import Dict, List, Callable, Tuple

from fuzzy_systems.core.membership_functions.free_shape_mf import FreeShapeMF
from fuzzy_systems.core.rules.fuzzy_rule_element import Antecedent, Consequent


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

        :param ant_act_func: A Tuple[Callable, str] where the callable is
        either a t-norm or a t-conorm operator and where the string is used
        for visualization purposes. Generally, FIS.AND_min or FIS.OR_max is used

        :param cons:a list of Consequent

        :param impl_func: Implication function i.e. the function f(a,b) where
        a is a scalar, the result value of the antecedents activation of this
        rule, and where b represents the membership function(s) of the
        consequent(s) used in the rule. This function will return an implicated
        membership function. Generally, min or product are used.
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
        This function will fuzzify crisp input values on each rule's antecedents

        :param crisp_inputs: the rule's antecedents crisps inputs values i.e. a
        user's/dataset sample input. Example crisp_inputs = {"temperature": 18,
        "sunshine": 55}

        :return: a list of fuzzified inputs (same size as the number of
        antecedents) for this particular rule
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

        impl_func = self._impl_func[0]
        implicated_consequents = defaultdict(list)

        for con in self._cons:
            # get the output variable's MF used by this specific consequent
            # in this rule. For example the MF of "warm" in the case of
            # the linguistic variable "temperature".
            ling_value = con.lv_name[con.lv_value]

            in_values = deepcopy(ling_value.in_values)  # FIXME deepcopy needed?
            mf_values = [impl_func([val, antecedents_activation]) for
                         val in ling_value.mf_values]

            # lv_name.name is the name of the linguistic variable, e.g.
            # "temperature"
            implicated_consequents[con.lv_name.name].append(
                FreeShapeMF(in_values, mf_values))

        return implicated_consequents

    def get_output_variable_names(self):
        return [con.lv_value.name for con in self.consequents]

    def __repr__(self):
        text = "IF ({}), THEN ({})"

        ants_text = " {} ".format(self._ant_act_func[1]).join(
            ["{} is {}".format(a.lv_name.name, a.lv_value) for a in
             self.antecedents])

        cons_text = " {} ".format(",").join(
            ["{} is {}".format(c.lv_name.name, c.lv_value) for c in
             self.consequents])

        return text.format(ants_text, cons_text)
