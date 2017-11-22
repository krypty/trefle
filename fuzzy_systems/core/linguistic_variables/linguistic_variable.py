from abc import ABCMeta
from typing import Dict

from fuzzy_systems.core.membership_functions.free_shape_mf import FreeShapeMF


class LinguisticVariable(metaclass=ABCMeta):
    """
    This class represents a linguistic variable (LV). Basically a LV
    has a name (e.g. "Temperature") and associated linguistic values
    (that basically contains a name (e.g. "Cold") and a membership function
    that represent it).
    """

    def __init__(self, name: str, ling_values_dict: Dict[str, FreeShapeMF]):
        """
        :param name: name of the linguistic variable (e.g. "Temperature")
        :param ling_values_dict: dict that contains the associated linguistic
        values for the linguistic variable. The dict's keys contains the name
        of the linguistic values (e.g. "Cold") and the values contains the
        membership function that represents it (i.e. an instance of
        FreeShapeMF)
        """
        self._name = name
        self._ling_values_dict = ling_values_dict
        self._in_range = self._compute_in_range()

    @property
    def name(self):
        return self._name

    @property
    def ling_values(self):
        return self._ling_values_dict

    @property
    def labels_name(self):
        return self._ling_values_dict.keys()

    @property
    def in_range(self):
        return self._in_range

    def __getitem__(self, ling_value: str):
        """
        Syntactic sugar to directly access to linguistic values given its name.
        Example: lv_temperature["Cold"].fuzzify(x)
        :param ling_value:
        :return: the linguistic value associated to the key ling_value
        """
        return self._ling_values_dict[ling_value]

    def __str__(self):
        return "Name: {}, values: {}".format(self.name,
                                             self._ling_values_dict.keys())

    def _compute_in_range(self):
        a = [[min(mf.in_values), max(mf.in_values)] for mf in
             self._ling_values_dict.values()]

        in_min, in_max = zip(*a)
        return min(in_min), max(in_max)
