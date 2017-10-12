import numpy as np

from core.fis.fis import FIS, OR_max, AND_min, COA_func, MIN
from core.linguistic_variables.linguistic_variable import LinguisticVariable
from core.membership_functions.lin_piece_wise_mf import LinPWMF
from core.rules.fuzzy_rule import FuzzyRule

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

    # lvs = [difficulty, importance, remaining_work, time_to_exam, priority]
    #
    # fig, axs = plt.subplots(len(lvs), figsize=(12, 8))
    #
    # for lv, ax in zip(lvs, axs):
    #     LinguisticVariableViewer(lv, ax)
    #
    # plt.show()
    #
    #
    # assert False

    rules = []

    # rules.append(FuzzyRule(
    #     ants=[(time_to_exam, 'very close'), (time_to_exam, 'close'),
    #           (importance, 'high')],
    #     ant_act_func=OR_max,
    #     cons=[(priority, 'high')],
    #     impl_func=np.min
    # ))

    # High priority rules
    rules.append(FuzzyRule(ant_act_func=OR_max,
                           ants=[(time_to_exam, 'very close'),
                                 (time_to_exam, 'close'), (importance, 'high')],
                           cons=[(priority, 'high')],
                           impl_func=MIN))

    rules.append(FuzzyRule(ant_act_func=AND_min,
                           ants=[(remaining_work, 'lot'),
                                 (difficulty, 'very hard')],
                           cons=[(priority, 'high')],
                           impl_func=MIN))

    rules.append(FuzzyRule(ant_act_func=AND_min,
                           ants=[(remaining_work, 'lot'), (difficulty, 'hard')],
                           cons=[(priority, 'high')],
                           impl_func=MIN))

    rules.append(FuzzyRule(ant_act_func=AND_min,
                           ants=[(remaining_work, 'medium'),
                                 (difficulty, 'very hard')],
                           cons=[(priority, 'high')],
                           impl_func=MIN))

    # Medium priority rules
    rules.append(FuzzyRule(ant_act_func=OR_max,
                           ants=[(time_to_exam, 'medium'),
                                 (importance, 'medium')],
                           cons=[(priority, 'medium')],
                           impl_func=MIN))

    rules.append(FuzzyRule(ant_act_func=AND_min,
                           ants=[(remaining_work, 'few'),
                                 (difficulty, 'very hard')],
                           cons=[(priority, 'medium')],
                           impl_func=MIN))

    rules.append(FuzzyRule(ant_act_func=AND_min,
                           ants=[(remaining_work, 'medium'),
                                 (difficulty, 'hard')],
                           cons=[(priority, 'medium')],
                           impl_func=MIN))

    rules.append(FuzzyRule(ant_act_func=AND_min,
                           ants=[(remaining_work, 'lot'),
                                 (difficulty, 'medium')],
                           cons=[(priority, 'medium')],
                           impl_func=MIN))

    rules.append(FuzzyRule(ant_act_func=AND_min,
                           ants=[(remaining_work, 'lot'), (difficulty, 'easy')],
                           cons=[(priority, 'medium')],
                           impl_func=MIN))

    # Low priority rules
    rules.append(FuzzyRule(ant_act_func=OR_max,
                           ants=[(time_to_exam, 'very far'),
                                 (time_to_exam, 'far'), (importance, 'low')],
                           cons=[(priority, 'low')],
                           impl_func=MIN))

    rules.append(FuzzyRule(ant_act_func=AND_min,
                           ants=[(remaining_work, 'lot'),
                                 (difficulty, 'very easy')],
                           cons=[(priority, 'low')],
                           impl_func=MIN))

    rules.append(FuzzyRule(ant_act_func=AND_min,
                           ants=[(remaining_work, 'medium'),
                                 (difficulty, 'medium')],
                           cons=[(priority, 'low')],
                           impl_func=MIN))

    rules.append(FuzzyRule(ant_act_func=AND_min,
                           ants=[(remaining_work, 'medium'),
                                 (difficulty, 'easy')],
                           cons=[(priority, 'low')],
                           impl_func=MIN))

    rules.append(FuzzyRule(ant_act_func=AND_min,
                           ants=[(remaining_work, 'medium'),
                                 (difficulty, 'very easy')],
                           cons=[(priority, 'low')],
                           impl_func=MIN))

    rules.append(FuzzyRule(ant_act_func=AND_min,
                           ants=[(remaining_work, 'few'), (difficulty, 'hard')],
                           cons=[(priority, 'low')],
                           impl_func=MIN))

    rules.append(FuzzyRule(ant_act_func=AND_min,
                           ants=[(remaining_work, 'few'),
                                 (difficulty, 'medium')],
                           cons=[(priority, 'low')],
                           impl_func=MIN))

    rules.append(FuzzyRule(ant_act_func=AND_min,
                           ants=[(remaining_work, 'few'), (difficulty, 'easy')],
                           cons=[(priority, 'low')],
                           impl_func=MIN))

    rules.append(FuzzyRule(ant_act_func=AND_min,
                           ants=[(remaining_work, 'few'),
                                 (difficulty, 'very easy')],
                           cons=[(priority, 'low')],
                           impl_func=MIN))

    fis = FIS(aggr_func=np.max, defuzz_func=COA_func, rules=rules)

    r = []
    # class 1
    r.append(fis.predict(crisp_values={'difficulty': 8, 'importance': 4,
                                       'remaining_work': 30,
                                       'time_to_exam': 4}))

    # class 2
    r.append(fis.predict(crisp_values={'difficulty': 4, 'importance': 6,
                                       'remaining_work': 50,
                                       'time_to_exam': 6}))

    # class 3
    r.append(fis.predict(crisp_values={'difficulty': 10, 'importance': 7,
                                       'remaining_work': 10,
                                       'time_to_exam': 10}))

    # class 3
    r.append(fis.predict(crisp_values={'difficulty': 2, 'importance': 3,
                                       'remaining_work': 80,
                                       'time_to_exam': 3}))

    # class 4
    r.append(fis.predict(crisp_values={'difficulty': 6, 'importance': 9,
                                       'remaining_work': 40,
                                       'time_to_exam': 13}))

    # class 5
    r.append(fis.predict(crisp_values={'difficulty': 4, 'importance': 7,
                                       'remaining_work': 10,
                                       'time_to_exam': 10}))

    # class 6
    r.append(fis.predict(crisp_values={'difficulty': 1, 'importance': 7,
                                       'remaining_work': 10,
                                       'time_to_exam': 10}))

    # class 7
    r.append(fis.predict(crisp_values={'difficulty': 10, 'importance': 10,
                                       'remaining_work': 10,
                                       'time_to_exam': 10}))

    r.append(fis.predict(
        crisp_values={'difficulty': 10, 'importance': 1, 'remaining_work': 10,
                      'time_to_exam': 10}))
    r.append(fis.predict(
        crisp_values={'difficulty': 10, 'importance': 7, 'remaining_work': 1,
                      'time_to_exam': 10}))
    r.append(fis.predict(
        crisp_values={'difficulty': 5, 'importance': 7, 'remaining_work': 1,
                      'time_to_exam': 10}))
    r.append(fis.predict(
        crisp_values={'difficulty': 10, 'importance': 7, 'remaining_work': 10,
                      'time_to_exam': 1}))
    r.append(fis.predict(
        crisp_values={'difficulty': 10, 'importance': 7, 'remaining_work': 10,
                      'time_to_exam': 30}))
    r.append(fis.predict(
        crisp_values={'difficulty': 10, 'importance': 10, 'remaining_work': 100,
                      'time_to_exam': 1}))
    r.append(fis.predict(
        crisp_values={'difficulty': 1, 'importance': 1, 'remaining_work': 0,
                      'time_to_exam': 30}))

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

    for res, exp_res in zip(r, a):
        res_val = res["priority"]
        exp_res_val = exp_res[1]
        diff = abs(res_val - exp_res_val)
        print("computed {:.3f} vs {:.3f} = {:.3f}".format(res_val, exp_res_val,
                                                          diff))


def predict(i, fis, input_values):
    predicted_values = fis.predict(input_values)
    print("class {}: {}".format(i, predicted_values))


if __name__ == '__main__':
    main()
