import numpy as np

from core.fis.fis import FIS, OR_max, COA_func
from core.linguistic_variables.linguistic_variable import LinguisticVariable
from core.membership_functions.lin_piece_wise_mf import LinPWMF
from core.rules.fuzzy_rule import FuzzyRule


def tip_problem():
    lv_quality = LinguisticVariable(name="quality", ling_values_dict={
        "poor": LinPWMF([0, 1], [5, 0]),
        "average": LinPWMF([0, 0], [5, 1], [10, 0]),
        "good": LinPWMF([5, 0], [10, 1])
    })

    lv_service = LinguisticVariable(name="service", ling_values_dict={
        "poor": LinPWMF([0, 1], [5, 0]),
        "average": LinPWMF([0, 0], [5, 1], [10, 0]),
        "good": LinPWMF([5, 0], [10, 1])
    })

    lv_tip = LinguisticVariable(name="tip", ling_values_dict={
        "low": LinPWMF([0, 1], [13, 0]),
        "medium": LinPWMF([0, 0], [13, 1], [25, 0]),
        "high": LinPWMF([13, 0], [25, 1])
    })

    r1 = FuzzyRule(
        ants=[
            (lv_quality, "poor"),
            (lv_service, "poor")
        ],
        ant_act_func=OR_max,
        cons=[
            (lv_tip, "low")
        ],
        impl_func=np.min
    )

    r2 = FuzzyRule(
        ants=[
            (lv_service, "average"),
        ],
        ant_act_func=OR_max,
        cons=[
            (lv_tip, "medium")
        ],
        impl_func=np.min
    )

    r3 = FuzzyRule(
        ants=[
            (lv_service, "good"),
            (lv_quality, "good")
        ],
        ant_act_func=OR_max,
        cons=[
            (lv_tip, "high")
        ],
        impl_func=np.min
    )

    fis = FIS(
        aggr_func=np.max,
        defuzz_func=COA_func,
        rules=[r1, r2, r3]
    )

    input_values = {'quality': 6.5, 'service': 9.8}
    predicted_value = fis.predict(input_values)["tip"]

    expected_value = 48.3
    print("predicted value: {}".format(predicted_value))
    print("expected  value: {}".format(expected_value))
    print("difference     : {}".format(expected_value - predicted_value))


if __name__ == '__main__':
    # import cProfile
    # cProfile.run("[resort_problem() for _ in range(1000)]")
    tip_problem()
