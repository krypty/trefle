import numpy as np

from pyfuge.fuzzy_systems.core.fis.fis import MIN, AND_min
from pyfuge.fuzzy_systems.core.fis.singleton_fis import SingletonFIS
from pyfuge.fuzzy_systems.core.linguistic_variables.linguistic_variable import \
    LinguisticVariable
from pyfuge.fuzzy_systems.core.linguistic_variables.three_points_lv import \
    ThreePointsLV
from pyfuge.fuzzy_systems.core.membership_functions.singleton_mf import \
    SingletonMF
from pyfuge.fuzzy_systems.core.rules.default_fuzzy_rule import DefaultFuzzyRule
from pyfuge.fuzzy_systems.core.rules.fuzzy_rule import FuzzyRule
from pyfuge.fuzzy_systems.core.rules.fuzzy_rule_element import Antecedent, \
    Consequent

"""
IRIS DATASET SOURCE: 
https://archive.ics.uci.edu/ml/machine-learning-databases/iris/
"""


def main():
    from time import time

    t0 = time()

    # Build FIS from Fig 3.9 of Carlos Peña's book
    lv_sl = ThreePointsLV(name="SL", p1=4.65, p2=4.65, p3=5.81)
    lv_sw = ThreePointsLV(name="SW", p1=2.68, p2=3.74, p3=4.61)
    lv_pl = ThreePointsLV(name="PL", p1=4.68, p2=5.26, p3=6.03)
    lv_pw = ThreePointsLV(name="PW", p1=0.39, p2=1.16, p3=2.03)

    yes_no = {
        "no": SingletonMF(0),
        "yes": SingletonMF(1)
    }
    lv_setosa = LinguisticVariable(name="setosa", ling_values_dict=yes_no)
    lv_virginica = LinguisticVariable(name="virginica", ling_values_dict=yes_no)
    lv_versicolor = LinguisticVariable(name="versicolor",
                                       ling_values_dict=yes_no)

    r1 = FuzzyRule(
        ants=[Antecedent(lv_pw, "low")],
        ant_act_func=AND_min,
        cons=[
            Consequent(lv_setosa, "yes"),
            Consequent(lv_versicolor, "no"),
            Consequent(lv_virginica, "no"),
        ],
        impl_func=MIN
    )

    r2 = FuzzyRule(
        ants=[
            Antecedent(lv_pl, "low"),
            Antecedent(lv_pw, "medium")
        ],
        ant_act_func=AND_min,
        cons=[
            Consequent(lv_setosa, "no"),
            Consequent(lv_versicolor, "yes"),
            Consequent(lv_virginica, "no"),
        ],
        impl_func=MIN
    )

    r3 = FuzzyRule(
        ants=[
            Antecedent(lv_sl, "high"),
            Antecedent(lv_sw, "medium"),
            Antecedent(lv_pl, "low"),
            Antecedent(lv_pw, "high"),
        ],
        ant_act_func=AND_min,
        cons=[
            Consequent(lv_setosa, "no"),
            Consequent(lv_versicolor, "yes"),
            Consequent(lv_virginica, "no"),
        ],
        impl_func=MIN
    )

    rules = [r1, r2, r3]
    dr = DefaultFuzzyRule(
        cons=[
            Consequent(lv_setosa, "no"),
            Consequent(lv_versicolor, "no"),
            Consequent(lv_virginica, "yes"),
        ],
        impl_func=MIN
    )

    fis = SingletonFIS(rules=rules, default_rule=dr)

    # Read Iris dataset
    iris_data = np.loadtxt(r'../../../datasets/iris.data', delimiter=",",
                           dtype="f8,f8,f8,f8,|U15")

    def check_prediction(predicted, expected, counter):
        if predicted == expected:
            counter += 1
        return counter

    yolo_preds = []

    n_correct_pred = 0
    for idx, sample in enumerate(iris_data):
        predicted_out = fis.predict({
            "SL": sample[0],
            "SW": sample[1],
            "PL": sample[2],
            "PW": sample[3]
        })
        # print("pred out", predicted_out)

        preds = sorted(predicted_out.items(), key=lambda p: p[1], reverse=True)
        print(preds)

        # setosa, versi, virgi
        yoloyolo = sorted(predicted_out.items(), key=lambda p: p[0],
                          reverse=False)

        a = [yoloyolo[0][1], yoloyolo[1][1], yoloyolo[2][1]]
        print(a)
        print("-" * 10)
        yolo_preds.append(a)

        # the max pred must reach at least 0.5 to be considered as a the true
        #  predicted output
        if preds[0][1] < 0.5:
            continue

        # e.g. "Iris-versicolor" --> "versicolor"
        expected_out_str = sample[4][5:]

        predicted_out_str = preds[0][0]

        n_correct_pred = check_prediction(predicted_out_str, expected_out_str,
                                          n_correct_pred)

        # print("[{}] expected {}, predicted {}".format(idx, expected_out_str,
        #                                               predicted_out_str))

    print((time() - t0) * 1000, "ms")

    print("pred OK: {}, total pred: {}".format(n_correct_pred, len(iris_data)))
    assert n_correct_pred == 149, "The book says this FIS must predict " \
                                  "correctly 149 cases "

    np.savetxt("/tmp/pyfuge_original.csv", np.array(yolo_preds), delimiter=";")


if __name__ == '__main__':
    main()
    # import cProfile
    # cProfile.run("main()")
