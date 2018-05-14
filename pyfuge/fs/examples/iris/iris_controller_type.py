import numpy as np

from pyfuge.fs.core.fis.fis import MIN, AND_min
from pyfuge.fs.core.fis.singleton_fis import SingletonFIS
from pyfuge.fs.core.lv.linguistic_variable import \
    LinguisticVariable
from pyfuge.fs.core.lv.three_points_lv import \
    ThreePointsLV
from pyfuge.fs.core.mf.singleton_mf import \
    SingletonMF
from pyfuge.fs.core.rules.default_fuzzy_rule import DefaultFuzzyRule
from pyfuge.fs.core.rules.fuzzy_rule import FuzzyRule
from pyfuge.fs.core.rules.fuzzy_rule_element import Antecedent, \
    Consequent

"""
IRIS DATASET SOURCE: 
https://archive.ics.uci.edu/ml/machine-learning-databases/iris/
"""


def main():
    # Build FIS from Fig 3.9 of Carlos Peña's book
    lv_sl = ThreePointsLV(name="SL", p1=5.68, p2=6.45, p3=7.10)
    lv_sw = ThreePointsLV(name="SW", p1=3.16, p2=3.16, p3=3.45)
    lv_pl = ThreePointsLV(name="PL", p1=1.19, p2=1.77, p3=6.03)
    lv_pw = ThreePointsLV(name="PW", p1=1.55, p2=1.65, p3=1.74)

    lv_output = LinguisticVariable(name="output", ling_values_dict={
        "setosa": SingletonMF(1),
        "versicolor": SingletonMF(2),
        "virginica": SingletonMF(3)
    })

    r1 = FuzzyRule(
        ants=[Antecedent(lv_pl, "high")],
        ant_act_func=AND_min,
        cons=[
            Consequent(lv_output, "virginica"),
        ],
        impl_func=MIN
    )

    r2 = FuzzyRule(
        ants=[
            Antecedent(lv_sw, "low"),
            Antecedent(lv_pw, "high")
        ],
        ant_act_func=AND_min,
        cons=[
            Consequent(lv_output, "virginica"),
        ],
        impl_func=MIN
    )

    r3 = FuzzyRule(
        ants=[
            Antecedent(lv_sl, "medium"),
            Antecedent(lv_pw, "medium")
        ],
        ant_act_func=AND_min,
        cons=[
            Consequent(lv_output, "setosa"),
        ],
        impl_func=MIN
    )

    rules = [r1, r2, r3]
    dr = DefaultFuzzyRule(
        cons=[
            Consequent(lv_output, "setosa"),
        ],
        impl_func=MIN
    )

    fis = SingletonFIS(rules=rules, default_rule=dr)

    # Read Iris dataset
    iris_data = np.loadtxt(r'../../../datasets/iris.data', delimiter=",",
                           dtype="f8,f8,f8,f8,|U15")

    dict_output = {
        "setosa": 1,
        "versicolor": 2,
        "virginica": 3
    }

    n_correct_pred = 0
    for idx, sample in enumerate(iris_data):
        predicted_out = fis.predict({
            "SL": sample[0],
            "SW": sample[1],
            "PL": sample[2],
            "PW": sample[3]
        })

        expected_out_str = sample[4][5:]

        print("predicted {}, expected {}".format(predicted_out["output"],
                                                 dict_output[expected_out_str]))
        out = int(predicted_out["output"] - dict_output[expected_out_str] + 0.5)
        if out == 0:
            n_correct_pred += 1

    print(n_correct_pred, len(iris_data))
    assert n_correct_pred == len(iris_data), "The book says this FIS make no " \
                                             "misclassification "


if __name__ == '__main__':
    main()
