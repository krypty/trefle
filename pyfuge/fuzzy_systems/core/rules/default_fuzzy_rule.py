from typing import List, Callable, Tuple

from pyfuge.fuzzy_systems.core.rules.fuzzy_rule import FuzzyRule
from pyfuge.fuzzy_systems.core.rules.fuzzy_rule_element import Consequent


class DefaultFuzzyRule(FuzzyRule):
    def __init__(self, cons: List[Consequent], impl_func: Tuple[Callable, str]):
        """
        Define a default rule for a fuzzy system. It behaves the same as
        FuzzyRule but it does not require to define antecedents
        neither an activation function

        :param cons: see FuzzyRule's docstring
        :param impl_func: see FuzzyRule's docstring
        """
        super(DefaultFuzzyRule, self).__init__(
            ants=[],
            ant_act_func=None,  # can set any other act func, it is not used
            cons=cons,
            impl_func=impl_func
        )

    def __repr__(self):
        text = "ELSE ({})"

        cons_text = " {} ".format(self._impl_func[1]).join(
            ["{} is {}".format(c.lv_name.name, c.lv_value) for c in
             self.consequents])

        return text.format(cons_text)
