import numpy as np

from pyfuge.fs.core.fis.fis import AND_min, OR_max, MIN, COA_func, \
    FIS
from pyfuge.fs.core.lv.linguistic_variable import \
    LinguisticVariable
from pyfuge.fs.core.mf.lin_piece_wise_mf import \
    LinPWMF
from pyfuge.fs.core.mf.triangular_mf import \
    TriangularMF
from pyfuge.fs.core.rules.fuzzy_rule import FuzzyRule
from pyfuge.fs.core.rules.fuzzy_rule_element import Antecedent, \
    Consequent
from pyfuge.fs.view.fis_viewer import FISViewer


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
        "high": LinPWMF([50, 0], [100, 1])
    })

    # lv_tourists = LinguisticVariable(name="tourists", ling_values_dict={
    #     "low": SingletonMF(0),
    #     "medium": SingletonMF(50),
    #     "high": SingletonMF(100)
    # })

    r1 = FuzzyRule(
        ants=[
            Antecedent(lv_temperature, "hot"),
            Antecedent(lv_sunshine, "sunny")
        ],
        ant_act_func=OR_max,
        cons=[
            Consequent(lv_tourists, "high")
        ],
        impl_func=MIN
    )

    r2 = FuzzyRule(
        ants=[
            Antecedent(lv_temperature, "warm"),
            Antecedent(lv_sunshine, "part_sunny")
        ],
        ant_act_func=AND_min,
        cons=[
            Consequent(lv_tourists, "medium")
        ],
        impl_func=MIN
    )

    r3 = FuzzyRule(
        ants=[
            Antecedent(lv_temperature, "cold"),
            Antecedent(lv_sunshine, "cloudy")
        ],
        ant_act_func=OR_max,
        cons=[
            Consequent(lv_tourists, "low"),
        ],
        impl_func=MIN
    )

    # fis = SingletonFIS(
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

    # View the FIS
    fisv = FISViewer(fis, figsize=(12, 10))
    describe_fis(fis)
    # fisv.save("/tmp/out.png")
    fisv.show()


def describe_fis(fis):
    [print(r) for r in fis.rules]
    if fis.default_rule is not None:
        print(fis.default_rule)


if __name__ == '__main__':
    # import cProfile
    # cProfile.run("[resort_problem() for _ in range(1000)]")
    resort_problem()
