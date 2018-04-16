from fuzzy_systems.core.linguistic_variables.linguistic_variable import \
    LinguisticVariable
from fuzzy_systems.core.membership_functions.lin_piece_wise_mf import LinPWMF


class TwoPointsPDLV(LinguisticVariable):
    """
    Syntactic sugar for simplified linguistic variable with only 2 points (p1 and
    p2) and fixed labels ("low", and "high").


      ^
      |
    1 |XXXXXXXXX                 XXXXXXXXXXX
      |        XX               XX
      |         XXX            XX
      |           XXX        XX
      |             XXX    XXX
      |               XX  XX
      |               XXXXX
      |             XXX    XXX
      |          XX           XX
      |        XX              XXX
    0 +------------------------------------>
              P<------ d ------>

    """

    def __init__(self, name, p, d):
        super(TwoPointsPDLV, self).__init__(name, ling_values_dict={
            "low": LinPWMF([p, 1], [d, 0]),
            "high": LinPWMF([p, 0], [d, 1])
        })
