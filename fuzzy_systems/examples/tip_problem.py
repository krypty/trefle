import numpy as np

from fuzzy_systems.core.fis.fis import OR_max, MIN, FIS, COA_func
from fuzzy_systems.core.linguistic_variables.linguistic_variable import \
    LinguisticVariable
from fuzzy_systems.core.membership_functions.lin_piece_wise_mf import LinPWMF
from fuzzy_systems.core.rules.fuzzy_rule import FuzzyRule
from fuzzy_systems.core.rules.fuzzy_rule_element import Antecedent, Consequent
from fuzzy_systems.view.fis_viewer import FISViewer


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
        # "low": SingletonMF(0),  # LinPWMF([0, 1], [13, 0]),
        # "medium": SingletonMF(25 // 2),  # LinPWMF([0, 0], [13, 1], [25, 0]),
        # "high": SingletonMF(25)  # LinPWMF([13, 0], [25, 1])

        "low": LinPWMF([0, 1], [13, 0]),
        "medium": LinPWMF([0, 0], [13, 1], [25, 0]),
        "high": LinPWMF([13, 0], [25, 1])
    })

    r1 = FuzzyRule(
        ants=[
            Antecedent(lv_quality, "poor"),
            Antecedent(lv_service, "poor")
        ],
        ant_act_func=OR_max,
        cons=[
            Consequent(lv_tip, "low"),
        ],
        impl_func=MIN
    )

    r2 = FuzzyRule(
        ants=[
            Antecedent(lv_service, "average"),
        ],
        ant_act_func=OR_max,
        cons=[
            Consequent(lv_tip, "medium"),
        ],
        impl_func=MIN
    )

    r3 = FuzzyRule(
        ants=[
            Antecedent(lv_service, "good"),
            Antecedent(lv_quality, "good")
        ],
        ant_act_func=OR_max,
        cons=[
            Consequent(lv_tip, "high"),
        ],
        impl_func=MIN
    )

    fis = FIS(
        rules=[r1, r2, r3],
        aggr_func=np.max,
        defuzz_func=COA_func
    )

    input_values = {'quality': 4.5, 'service': 3}
    predicted_value = fis.predict(input_values)["tip"]

    expected_value = 48.3
    print("predicted value: {}".format(predicted_value))
    print("expected  value: {}".format(expected_value))
    print("difference     : {}".format(expected_value - predicted_value))

    fisv = FISViewer(fis)
    fisv.show()


if __name__ == '__main__':
    # import cProfile
    # cProfile.run("[resort_problem() for _ in range(1000)]")
    tip_problem()
