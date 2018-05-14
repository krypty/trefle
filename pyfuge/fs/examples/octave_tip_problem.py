import numpy as np

from pyfuge.fs.core.fis.fis import COA_func, MIN, FIS, AND_min
from pyfuge.fs.core.linguistic_variables.linguistic_variable import \
    LinguisticVariable
from pyfuge.fs.core.membership_functions.gaussian_mf import \
    GaussianMF
from pyfuge.fs.core.membership_functions.lin_piece_wise_mf import \
    LinPWMF
from pyfuge.fs.core.rules.fuzzy_rule import FuzzyRule
from pyfuge.fs.core.rules.fuzzy_rule_element import Antecedent, \
    Consequent
from pyfuge.fs.view.fis_viewer import FISViewer


def tip_problem():
    lv_quality = LinguisticVariable(name="Food-Quality", ling_values_dict={
        "bad": LinPWMF([3, 1], [7, 0]),
        "good": LinPWMF([3, 0], [7, 1])
    })

    lv_service = LinguisticVariable(name="service", ling_values_dict={
        "bad": LinPWMF([3, 1], [7, 0]),
        "good": LinPWMF([3, 0], [7, 1])
    })

    lv_tip = LinguisticVariable(name="tip", ling_values_dict={
        "ten": GaussianMF(10, 2),
        "fifteen": GaussianMF(15, 2),
        "twenty": GaussianMF(20, 2)
    })

    lv_tip_plus_check = LinguisticVariable(
        name="tip_plus_check",
        ling_values_dict={
            "plus_ten": GaussianMF(1.10, 0.02),
            "plus_fifteen": GaussianMF(1.15, 0.02),
            "plus_twenty": GaussianMF(1.20, 0.02)
        })

    r1 = FuzzyRule(
        ants=[
            Antecedent(lv_quality, "bad"),
            Antecedent(lv_service, "bad")
        ],
        ant_act_func=AND_min,
        cons=[
            Consequent(lv_tip, "ten"),
            Consequent(lv_tip_plus_check, "plus_ten")
        ],
        impl_func=MIN
    )

    r2 = FuzzyRule(
        ants=[
            Antecedent(lv_quality, "bad"),
            Antecedent(lv_service, "good"),
        ],
        ant_act_func=AND_min,
        cons=[
            Consequent(lv_tip, "fifteen"),
            Consequent(lv_tip_plus_check, "plus_fifteen"),
        ],
        impl_func=MIN
    )

    r3 = FuzzyRule(
        ants=[
            Antecedent(lv_quality, "good"),
            Antecedent(lv_service, "bad")
        ],
        ant_act_func=AND_min,
        cons=[
            Consequent(lv_tip, "fifteen"),
            Consequent(lv_tip_plus_check, "plus_fifteen"),
        ],
        impl_func=MIN
    )

    r4 = FuzzyRule(
        ants=[
            Antecedent(lv_quality, "good"),
            Antecedent(lv_service, "good")
        ],
        ant_act_func=AND_min,
        cons=[
            Consequent(lv_tip, "twenty"),
            Consequent(lv_tip_plus_check, "plus_twenty"),
        ],
        impl_func=MIN
    )

    fis = FIS(
        aggr_func=np.max,
        defuzz_func=COA_func,
        rules=[r1, r2, r3, r4]
    )

    input_values = {'Food-Quality': 4, 'service': 6}
    predicted_values = fis.predict(input_values)
    print(predicted_values)

    # expected_value = 48.3
    # print("predicted value: {}".format(predicted_value))
    # print("expected  value: {}".format(expected_value))
    # print("difference     : {}".format(expected_value - predicted_value))

    fisv = FISViewer(fis)
    fisv.show()


if __name__ == '__main__':
    # import cProfile
    # cProfile.run("[resort_problem() for _ in range(1000)]")
    tip_problem()
