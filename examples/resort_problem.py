import numpy as np

from core.fis.fis import FIS
from core.linguistic_variables.linguistic_variable import LinguisticVariable
from core.membership_functions.lin_piece_wise_mf import LinPWMF
from core.membership_functions.triangular_mf import TriangularMF
from core.rules.fuzzy_rule import FuzzyRule

COA_func = lambda v, m: np.sum(np.multiply(v, m)) / np.sum(m)
OR_max = np.max
AND_min = np.min


def resort_problem():
    lv_temperature = LinguisticVariable(name="temperature", ling_values_dict={
        "cold": LinPWMF([17, 1], [20, 0]),
        "warm": LinPWMF([17, 0], [20, 1], [26, 1], [29, 0]),
        "hot": LinPWMF([26, 0], [29, 1])
    })

    lv_sunshine = LinguisticVariable(name="sunshine", ling_values_dict={
        "cloudy": LinPWMF([30, 1], [50, 0]),
        "part_sunny": TriangularMF(p_min=30, p_mid=50, p_max=100),
        "sunny": LinPWMF([50, 0], [100, 1])
    })

    lv_tourists = LinguisticVariable(name="tourists", ling_values_dict={
        "low": LinPWMF([0, 1], [50, 0]),
        "medium": TriangularMF(p_min=0, p_mid=50, p_max=100),
        "high": LinPWMF([0, 0], [50, 0], [100, 1])
    })

    r1 = FuzzyRule(
        ants=[
            (lv_temperature, "hot"),
            (lv_sunshine, "sunny")
        ],
        ant_act_func=OR_max,
        cons=[
            (lv_tourists, "high")
        ],
        impl_func=np.min
    )

    r2 = FuzzyRule(
        ants=[
            (lv_temperature, "warm"),
            (lv_sunshine, "part_sunny")
        ],
        ant_act_func=AND_min,
        cons=[
            (lv_tourists, "medium")
        ],
        impl_func=np.min
    )

    r3 = FuzzyRule(
        ants=[
            (lv_temperature, "cold"),
            (lv_sunshine, "cloudy")
        ],
        ant_act_func=OR_max,
        cons=[
            (lv_tourists, "low")
        ],
        impl_func=np.min
    )

    fis = FIS(
        aggr_func=np.max,
        defuzz_func=COA_func,
        rules=[r1, r2, r3]
    )

    input_values = {'temperature': 19, 'sunshine': 60}
    predicted_value = fis.predict(input_values)["tourists"]

    expected_value = 48.3
    print("predicted value: {}".format(predicted_value))
    print("expected  value: {}".format(expected_value))
    print("difference     : {}".format(expected_value - predicted_value))


if __name__ == '__main__':
    # import cProfile
    # cProfile.run("[resort_problem() for _ in range(1000)]")
    resort_problem()
