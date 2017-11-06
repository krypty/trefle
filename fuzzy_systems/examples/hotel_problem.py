import numpy as np

from core.fis.fis import FIS, OR_max, COA_func, MIN
from core.linguistic_variables.linguistic_variable import LinguisticVariable
from core.membership_functions.lin_piece_wise_mf import LinPWMF
from core.rules.fuzzy_rule import FuzzyRule
from core.rules.fuzzy_rule_element import Consequent, Antecedent
from view.fis_viewer import FISViewer
from view.lv_viewer import LinguisticVariableViewer
from view.mf_viewer import MembershipFunctionViewer


def hotel_problem():
    lv_service = LinguisticVariable(name="service", ling_values_dict={
        "bad": LinPWMF([0, 1], [4, 0]),
        "average": LinPWMF([4, 0], [5, 1], [6, 0]),
        "good": LinPWMF([6, 0], [7, 1], [8, 0]),
        "very good": LinPWMF([7, 0], [9, 1], [10, 1])
    })

    # LinguisticVariableViewer(lv_service).show()

    lv_recommendation = LinguisticVariable(
        name="recommendation",
        ling_values_dict={
            "low": LinPWMF([0, 0], [100, 1]),
            "high": LinPWMF([0, 1], [100, 0])
        })

    r1 = FuzzyRule(
        ants=[
            Antecedent(lv_service, "good"),
            Antecedent(lv_service, "very good")
        ],
        ant_act_func=OR_max,
        cons=[
            Consequent(lv_recommendation, "high")
        ],
        impl_func=MIN
    )

    print(r1)

    fis = FIS(
        aggr_func=np.max,
        defuzz_func=COA_func,
        rules=[r1]
    )

    input_values = {'service': 7.5}
    predicted_value = fis.predict(input_values)["recommendation"]

    FISViewer(fis).show()

    # cons = fis.last_implicated_consequents
    # MembershipFunctionViewer(cons["recommendation"][0]).show()

    expected_value = 0
    print("predicted value: {}".format(predicted_value))
    print("expected  value: {}".format(expected_value))
    print("difference     : {}".format(expected_value - predicted_value))


if __name__ == '__main__':
    # import cProfile
    # cProfile.run("[resort_problem() for _ in range(1000)]")
    hotel_problem()
