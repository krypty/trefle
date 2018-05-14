import numpy as np

from pyfuge.fs.core.fis.fis import FIS, OR_max, COA_func, MIN, \
    AND_min
from pyfuge.fs.core.lv.linguistic_variable import \
    LinguisticVariable
from pyfuge.fs.core.mf.lin_piece_wise_mf import \
    LinPWMF
from pyfuge.fs.core.rules.default_fuzzy_rule import DefaultFuzzyRule
from pyfuge.fs.core.rules.fuzzy_rule import FuzzyRule
from pyfuge.fs.core.rules.fuzzy_rule_element import Consequent, \
    Antecedent
from pyfuge.fs.view.fis_viewer import FISViewer


def hotel_problem():
    lv_room_quality = LinguisticVariable(name="service", ling_values_dict={
        "bad": LinPWMF([0, 1], [3, 0]),
        "average": LinPWMF([3, 0], [4.5, 1], [5.5, 0]),
        "good": LinPWMF([5.5, 0], [7, 1], [9.5, 0]),
        "very good": LinPWMF([7, 0], [12, 1])
    })

    lv_food_quality = LinguisticVariable(name="food", ling_values_dict={
        "disgusting": LinPWMF([0, 1], [8, 0]),
        "ok": LinPWMF([4, 0], [8, 1], [15, 0]),
        "delicious": LinPWMF([12, 0], [20, 1]),
    })

    lv_noise = LinguisticVariable(name="noise", ling_values_dict={
        "quiet": LinPWMF([30, 1], [50, 0]),
        "noisy": LinPWMF([50, 0], [80, 1]),
    })

    lv_recommendation = LinguisticVariable(
        name="recommendation",
        ling_values_dict={
            "low": LinPWMF([0, 1], [2.5, 0]),
            "medium": LinPWMF([2.5, 0], [3.5, 1], [4.5, 1], [5, 0]),
            "high": LinPWMF([4.5, 0], [6, 1])
        })

    lv_going_back = LinguisticVariable(
        name="going back",
        ling_values_dict={
            "no": LinPWMF([0, 1], [10, 0]),
            "yes": LinPWMF([0, 0], [10, 1])
        }
    )

    r1 = FuzzyRule(
        ants=[
            Antecedent(lv_room_quality, "very good"),
            Antecedent(lv_food_quality, "ok"),
            Antecedent(lv_food_quality, "delicious"),
        ],
        ant_act_func=OR_max,
        cons=[
            Consequent(lv_recommendation, "high"),
            Consequent(lv_going_back, "yes")
        ],
        impl_func=MIN
    )

    r2 = FuzzyRule(
        ants=[
            Antecedent(lv_room_quality, "good"),
            Antecedent(lv_noise, "quiet")
        ],
        ant_act_func=AND_min,
        cons=[
            Consequent(lv_recommendation, "medium"),
            Consequent(lv_going_back, "yes"),
        ],
        impl_func=MIN
    )

    r3 = FuzzyRule(
        ants=[
            Antecedent(lv_room_quality, "bad"),
            Antecedent(lv_food_quality, "disgusting"),
            Antecedent(lv_noise, "noisy"),
        ],
        ant_act_func=OR_max,
        cons=[
            Consequent(lv_recommendation, "low"),
            Consequent(lv_going_back, "no"),
        ],
        impl_func=MIN
    )

    rd = DefaultFuzzyRule(
        cons=[
            Consequent(lv_recommendation, "medium"),
            Consequent(lv_going_back, "no"),
        ],
        impl_func=MIN
    )

    print(r1)

    fis = FIS(
        aggr_func=np.max,
        defuzz_func=COA_func,
        rules=[r1, r2, r3],
        default_rule=rd
    )

    input_values = {'service': 9, "food": 17, "noise": 35}
    # fis.predict(input_values)

    FISViewer(fis).show()

    # cons = fis.last_implicated_consequents
    # MembershipFunctionViewer(cons["recommendation"][0]).show()


if __name__ == '__main__':
    # import cProfile
    # cProfile.run("[resort_problem() for _ in range(1000)]")
    hotel_problem()
