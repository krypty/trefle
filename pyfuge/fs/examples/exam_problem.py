import numpy as np

from pyfuge.fs.core.fis.fis import FIS, OR_max, AND_min, COA_func, \
    MIN
from pyfuge.fs.core.lv.linguistic_variable import \
    LinguisticVariable
from pyfuge.fs.core.mf.lin_piece_wise_mf import \
    LinPWMF
from pyfuge.fs.core.rules.fuzzy_rule import FuzzyRule
from pyfuge.fs.core.rules.fuzzy_rule_element import Antecedent, \
    Consequent
from pyfuge.fs.view.fis_surface import show_surface

"""
The outputs come from external_references/SANTAMARIA_LFA_LABO4-5/Labo-5.
They are not guaranteed to be 100% accurate but they give a good start.

Therefore, the reference presents a bug that gives small errors on implication
because the way it was implemented (use of res=XXX i.e. create points at 
regular interval (range(a,b,step)) instead of np.linspace(a, b, n_pts)). This
causes some "breaks/cuts" in slopes of fuzzy sets. 
"""

i = 0


def inc_i():
    global i
    i += 1
    return i


def main():
    difficulty = LinguisticVariable("difficulty", ling_values_dict={
        "very easy": LinPWMF([0, 1], [3, 0]),
        "easy": LinPWMF([0, 0], [3, 1], [5, 0]),
        "medium": LinPWMF([3, 0], [5, 1], [6, 1], [8, 0]),
        "hard": LinPWMF([6, 0], [8, 1], [10, 0]),
        "very hard": LinPWMF([8, 0], [10, 1])
    })

    importance = LinguisticVariable("importance", ling_values_dict={
        "low": LinPWMF([1, 1], [5, 0]),
        "medium": LinPWMF([1, 0], [5, 1], [6, 1], [10, 0]),
        "high": LinPWMF([6, 0], [10, 1]),
    })

    remaining_work = LinguisticVariable("remaining_work", ling_values_dict={
        "few": LinPWMF([0, 1], [50, 0]),
        "medium": LinPWMF([0, 0], [50, 1], [100, 0]),
        "lot": LinPWMF([50, 0], [100, 1]),
    })

    time_to_exam = LinguisticVariable("time_to_exam", ling_values_dict={
        "very close": LinPWMF([0, 1], [8, 0]),
        "close": LinPWMF([0, 0], [8, 1], [14, 0]),
        "medium": LinPWMF([8, 0], [14, 1], [16, 1], [23, 0]),
        "far": LinPWMF([16, 0], [23, 1], [30, 0]),
        "very far": LinPWMF([23, 0], [30, 1]),
    })

    priority = LinguisticVariable("priority", ling_values_dict={
        "low": LinPWMF([0, 1], [0.5, 0]),
        "medium": LinPWMF([0, 0], [0.5, 1], [1, 0]),
        "high": LinPWMF([0.5, 0], [1, 1]),
    })

    rules = []

    # High priority rules
    rules.append(FuzzyRule(ant_act_func=OR_max,
                           ants=[Antecedent(time_to_exam, 'very close'),
                                 Antecedent(time_to_exam, 'close'),
                                 Antecedent(importance, 'high')],
                           cons=[Consequent(priority, 'high')],
                           impl_func=MIN))

    rules.append(FuzzyRule(ant_act_func=AND_min,
                           ants=[Antecedent(remaining_work, 'lot'),
                                 Antecedent(difficulty, 'very hard')],
                           cons=[Consequent(priority, 'high')],
                           impl_func=MIN))

    rules.append(FuzzyRule(ant_act_func=AND_min,
                           ants=[Antecedent(remaining_work, 'lot'),
                                 Antecedent(difficulty, 'hard')],
                           cons=[Consequent(priority, 'high')],
                           impl_func=MIN))

    rules.append(FuzzyRule(ant_act_func=AND_min,
                           ants=[Antecedent(remaining_work, 'medium'),
                                 Antecedent(difficulty, 'very hard')],
                           cons=[Consequent(priority, 'high')],
                           impl_func=MIN))

    # Medium priority rules
    rules.append(FuzzyRule(ant_act_func=OR_max,
                           ants=[Antecedent(time_to_exam, 'medium'),
                                 Antecedent(importance, 'medium')],
                           cons=[Consequent(priority, 'medium')],
                           impl_func=MIN))

    rules.append(FuzzyRule(ant_act_func=AND_min,
                           ants=[Antecedent(remaining_work, 'few'),
                                 Antecedent(difficulty, 'very hard')],
                           cons=[Consequent(priority, 'medium')],
                           impl_func=MIN))

    rules.append(FuzzyRule(ant_act_func=AND_min,
                           ants=[Antecedent(remaining_work, 'medium'),
                                 Antecedent(difficulty, 'hard')],
                           cons=[Consequent(priority, 'medium')],
                           impl_func=MIN))

    rules.append(FuzzyRule(ant_act_func=AND_min,
                           ants=[Antecedent(remaining_work, 'lot'),
                                 Antecedent(difficulty, 'medium')],
                           cons=[Consequent(priority, 'medium')],
                           impl_func=MIN))

    rules.append(FuzzyRule(ant_act_func=AND_min,
                           ants=[Antecedent(remaining_work, 'lot'),
                                 Antecedent(difficulty, 'easy')],
                           cons=[Consequent(priority, 'medium')],
                           impl_func=MIN))

    # Low priority rules
    rules.append(FuzzyRule(ant_act_func=OR_max,
                           ants=[Antecedent(time_to_exam, 'very far'),
                                 Antecedent(time_to_exam, 'far'),
                                 Antecedent(importance, 'low')],
                           cons=[Consequent(priority, 'low')],
                           impl_func=MIN))

    rules.append(FuzzyRule(ant_act_func=AND_min,
                           ants=[Antecedent(remaining_work, 'lot'),
                                 Antecedent(difficulty, 'very easy')],
                           cons=[Consequent(priority, 'low')],
                           impl_func=MIN))

    rules.append(FuzzyRule(ant_act_func=AND_min,
                           ants=[Antecedent(remaining_work, 'medium'),
                                 Antecedent(difficulty, 'medium')],
                           cons=[Consequent(priority, 'low')],
                           impl_func=MIN))

    rules.append(FuzzyRule(ant_act_func=AND_min,
                           ants=[Antecedent(remaining_work, 'medium'),
                                 Antecedent(difficulty, 'easy')],
                           cons=[Consequent(priority, 'low')],
                           impl_func=MIN))

    rules.append(FuzzyRule(ant_act_func=AND_min,
                           ants=[Antecedent(remaining_work, 'medium'),
                                 Antecedent(difficulty, 'very easy')],
                           cons=[Consequent(priority, 'low')],
                           impl_func=MIN))

    rules.append(FuzzyRule(ant_act_func=AND_min,
                           ants=[Antecedent(remaining_work, 'few'),
                                 Antecedent(difficulty, 'hard')],
                           cons=[Consequent(priority, 'low')],
                           impl_func=MIN))

    rules.append(FuzzyRule(ant_act_func=AND_min,
                           ants=[Antecedent(remaining_work, 'few'),
                                 Antecedent(difficulty, 'medium')],
                           cons=[Consequent(priority, 'low')],
                           impl_func=MIN))

    rules.append(FuzzyRule(ant_act_func=AND_min,
                           ants=[Antecedent(remaining_work, 'few'),
                                 Antecedent(difficulty, 'easy')],
                           cons=[Consequent(priority, 'low')],
                           impl_func=MIN))

    rules.append(FuzzyRule(ant_act_func=AND_min,
                           ants=[Antecedent(remaining_work, 'few'),
                                 Antecedent(difficulty, 'very easy')],
                           cons=[Consequent(priority, 'low')],
                           impl_func=MIN))

    fis = FIS(aggr_func=np.max, defuzz_func=COA_func, rules=rules)

    results = []
    # class 1
    fis_predict(results, fis, crisp_values={'difficulty': 8, 'importance': 4,
                                            'remaining_work': 30,
                                            'time_to_exam': 4})

    # class 2
    fis_predict(results, fis, crisp_values={'difficulty': 4, 'importance': 6,
                                            'remaining_work': 50,
                                            'time_to_exam': 6})

    # class 3
    fis_predict(results, fis, crisp_values={'difficulty': 10, 'importance': 7,
                                            'remaining_work': 10,
                                            'time_to_exam': 10})

    # class 3
    fis_predict(results, fis, crisp_values={'difficulty': 2, 'importance': 3,
                                            'remaining_work': 80,
                                            'time_to_exam': 3})

    # class 4
    fis_predict(results, fis, crisp_values={'difficulty': 6, 'importance': 9,
                                            'remaining_work': 40,
                                            'time_to_exam': 13})

    # class 5
    fis_predict(results, fis, crisp_values={'difficulty': 4, 'importance': 7,
                                            'remaining_work': 10,
                                            'time_to_exam': 10})

    # class 6
    fis_predict(results, fis, crisp_values={'difficulty': 1, 'importance': 7,
                                            'remaining_work': 10,
                                            'time_to_exam': 10})

    # class 7
    fis_predict(results, fis, crisp_values={'difficulty': 10, 'importance': 10,
                                            'remaining_work': 10,
                                            'time_to_exam': 10})

    fis_predict(results, fis,
                crisp_values={'difficulty': 10, 'importance': 1,
                              'remaining_work': 10,
                              'time_to_exam': 10})
    fis_predict(results, fis,
                crisp_values={'difficulty': 10, 'importance': 7,
                              'remaining_work': 1,
                              'time_to_exam': 10})
    fis_predict(results, fis,
                crisp_values={'difficulty': 5, 'importance': 7,
                              'remaining_work': 1,
                              'time_to_exam': 10})
    fis_predict(results, fis,
                crisp_values={'difficulty': 10, 'importance': 7,
                              'remaining_work': 10,
                              'time_to_exam': 1})
    fis_predict(results, fis,
                crisp_values={'difficulty': 10, 'importance': 7,
                              'remaining_work': 10,
                              'time_to_exam': 30})
    fis_predict(results, fis,
                crisp_values={'difficulty': 10, 'importance': 10,
                              'remaining_work': 100,
                              'time_to_exam': 1})
    fis_predict(results, fis,
                crisp_values={'difficulty': 1, 'importance': 1,
                              'remaining_work': 0,
                              'time_to_exam': 30})

    # Example of using show_surface() with >2 input variables
    other_labels = {"time_to_exam": 4, "remaining_work": 80}
    show_surface(fis, title="Car Problem Mamdani",
                 x_label="difficulty", y_label="importance", z_label="priority",
                 n_pts=15, x_range=(0, 10), y_range=(0, 10), z_range=(0, 1),
                 other_labels=other_labels)

    other_labels = {"time_to_exam": 4, "remaining_work": 20}
    show_surface(fis, title="Car Problem Mamdani",
                 x_label="difficulty", y_label="importance", z_label="priority",
                 n_pts=15, x_range=(0, 10), y_range=(0, 10), z_range=(0, 1),
                 other_labels=other_labels)

    a = [
        (1, 0.530666027791088),
        (2, 0.530409356725146),
        (3, 0.594314159292035),
        (4, 0.53696682464455),
        (5, 0.494785407725322),
        (6, 0.526150121065375),
        (7, 0.483135657700398),
        (8, 0.620913016656385),
        (9, 0.472325860003423),
        (10, 0.591578947368421),
        (11, 0.471734475374732),
        (12, 0.620913016656385),
        (13, 0.405296547654175),
        (14, 0.866666666666667),
        (15, 0.133333333333333)
    ]

    for res, exp_res in zip(results, a):
        res_val = res["priority"]
        exp_res_val = exp_res[1]
        diff = abs(res_val - exp_res_val)
        print("computed {:.3f} vs {:.3f} = {:.3f}".format(res_val, exp_res_val,
                                                          diff))


def fis_predict(results, fis, crisp_values):
    res = fis.predict(crisp_values)
    # fisv = FISViewer(fis)
    # fisv.save("/tmp/out.png")
    # fisv.show()
    # input("Press enter...")
    results.append(res)


def predict(i, fis, input_values):
    predicted_values = fis.predict(input_values)
    print("class {}: {}".format(i, predicted_values))


if __name__ == '__main__':
    main()
