from fuzzy_systems.core.rules.fuzzy_rule import FuzzyRule


class DefaultFuzzyRule(FuzzyRule):
    def __init__(self, cons, impl_func):
        super(DefaultFuzzyRule, self).__init__(
            ants=[],
            ant_act_func=None,  # can set any other act func, it is not use
            cons=cons,
            impl_func=impl_func
        )

    def __repr__(self):
        text = "ELSE ({})"

        cons_text = " {} ".format(self._impl_func[1]).join(
            ["{} is {}".format(c.lv_name.name, c.lv_value) for c in
             self.consequents])

        return text.format(cons_text)
