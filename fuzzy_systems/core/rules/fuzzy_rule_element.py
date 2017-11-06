from fuzzy_systems.core.linguistic_variables.linguistic_variable import \
    LinguisticVariable


class FuzzyRuleElement:
    def __init__(self, lv_name: LinguisticVariable, lv_value: str,
                 is_not=False):
        """
        Define a fuzzy rule element that can be either an antecedent or a
        consequent
        :param lv_name: Linguistic variable name. e.g. "temperature"
        :param lv_value: Linguistic variable value. e.g. "cold"
        :param is_not: set it to True to indicate a not condition. e.g.
        is_not=True --> "temperature is NOT cold"
        """
        self._lv_name = lv_name
        self._lv_value = lv_value
        self._is_not = is_not

    @property
    def lv_name(self):
        return self._lv_name

    @property
    def lv_value(self):
        return self._lv_value

    @property
    def is_not(self):
        return self._is_not


class Antecedent(FuzzyRuleElement):
    """
    Syntactic sugar for FuzzyRuleElement
    """
    pass


class Consequent(FuzzyRuleElement):
    """
    Syntactic sugar for FuzzyRuleElement

    Limitations:

    * a consequent cannot be expressed as a NOT fuzzy rule element. e.g.
    "THEN my_consequent is NOT something" is considered invalid

    """

    def __init__(self, lv_name: LinguisticVariable, lv_value: str):
        super(Consequent, self).__init__(lv_name, lv_value, is_not=False)
