import numpy as np

from core.fis.fis import MIN, AND_min, COA_func
from core.fis.singleton_fis import SingletonFIS
from core.linguistic_variables.linguistic_variable import LinguisticVariable
from core.membership_functions.lin_piece_wise_mf import LinPWMF
from core.membership_functions.singleton_mf import SingletonMF
from core.rules.DefaultFuzzyRule import DefaultFuzzyRule
from core.rules.fuzzy_rule import FuzzyRule
from core.rules.fuzzy_rule_element import Antecedent, Consequent
from view.fis_viewer import FISViewer


class ThreePointsLV(LinguisticVariable):
    """
    Syntactic sugar for simplified linguistic variable with only 3 points (p1,
    p2 and p3) and fixed labels ("low", "medium" and "high").


      ^
      | low      medium           high
    1 |XXXXX       X          XXXXXXXXXXXX
      |     X     X  X       XX
      |      X   X    X    XX
      |       X X      XX X
      |       XX        XXX
      |      X  X     XX   XX
      |     X    X XX       XX
      |    X       X          XX
    0 +-------------------------------------->
           p1     p2          p3


    """

    def __init__(self, name, p1, p2, p3):
        assert p1 <= p2 <= p3, "points must be increasing values"
        super(ThreePointsLV, self).__init__(name, ling_values_dict={
            "low": LinPWMF([p1, 1], [p2, 0]),
            "medium": LinPWMF([p1, 0], [p2, 1], [p3, 0]),
            "high": LinPWMF([p2, 0], [p3, 1])
        })


def main():
    # Build FIS from Fig 3.9 of Carlos Peña's book
    lv_sl = ThreePointsLV(name="SL", p1=4.65, p2=4.65, p3=5.81)
    lv_sw = ThreePointsLV(name="SW", p1=2.68, p2=3.74, p3=4.61)
    lv_pl = ThreePointsLV(name="PL", p1=4.68, p2=5.26, p3=6.03)
    lv_pw = ThreePointsLV(name="PW", p1=0.39, p2=1.16, p3=2.03)

    lv_setosa = LinguisticVariable(name="setosa", ling_values_dict={
        "no": SingletonMF(0),
        "yes": SingletonMF(1)
    })
    lv_virginica = LinguisticVariable(name="virginica", ling_values_dict={
        "no": SingletonMF(0),
        "yes": SingletonMF(1)
    })
    lv_versicolor = LinguisticVariable(name="versicolor", ling_values_dict={
        "no": SingletonMF(0),
        "yes": SingletonMF(1)
    })

    # LinguisticVariableViewer(lv_sw)\
    #     .fuzzify(1)\
    #     .fuzzify(2)\
    #     .fuzzify(3.16)\
    #     .fuzzify(3.22)\
    #     .fuzzify(4.2)\
    #     .show()

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

    fis = SingletonFIS(
        rules=rules,
        default_rule=dr,
        aggr_func=AND_min,
        defuzz_func=COA_func,
    )

    # Read Iris dataset
    iris_data = np.loadtxt('iris.data', delimiter=",",
                           dtype="f8,f8,f8,f8,|U15")

    def check_prediction(predicted, expected, counter):
        if predicted == expected:
            counter += 1
        return counter

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

        # the max pred must reach at least 0.5 to be considered as a the true
        #  predicted output
        if preds[0][1] < 0.5:
            continue

        # e.g. "Iris-versicolor" --> "versicolor"
        expected_out_str = sample[4][5:]

        predicted_out_str = preds[0][0]

        n_correct_pred = check_prediction(predicted_out_str, expected_out_str,
                                          n_correct_pred)

        print("[{}] expected {}, predicted {}".format(idx,expected_out_str,
                                                 predicted_out_str))

    print("pred OK: {}, total pred: {}".format(n_correct_pred, len(iris_data)))
    assert n_correct_pred == 149, "The book says this FIS must predict " \
                                  "correctly 149 cases "


if __name__ == '__main__':
    main()
    # import cProfile
    # cProfile.run("main()")
