import numpy as np

from core.fis.fis import FIS, COA_func
from core.linguistic_variables.linguistic_variable import \
    LinguisticVariable as LVar
from core.membership_functions.lin_piece_wise_mf import LinPWMF
from core.rules.fuzzy_rule import FuzzyRule

AND_min = np.min
MIN = np.min


def car_accel_problem():
    car_speed = LVar(name='speed', ling_values_dict={
        'slow': LinPWMF([-20, 1], [0, 0]),
        'ok': LinPWMF([-20, 0], [0, 1], [15, 0]),
        'fast': LinPWMF([0, 0], [15, 1])
    })

    car_acc = LVar(name='acceleration', ling_values_dict={
        'slowing': LinPWMF([-20, 1], [0, 0]),
        'constant': LinPWMF([-20, 0], [0, 1], [20, 0]),
        'rising': LinPWMF([0, 0], [20, 1])
    })

    car_pedal = LVar(name='pedal', ling_values_dict={
        'release': LinPWMF([0, 1], [50, 0]),
        'nothing': LinPWMF([0, 0], [50, 1], [100, 0]),
        'push': LinPWMF([50, 0], [100, 1])
    })

    car_rules = [
        FuzzyRule(ant_act_func=AND_min,
                  ants=[(car_speed, 'fast'), (car_acc, 'constant')],
                  cons=[(car_pedal, 'release')], impl_func=MIN),

        FuzzyRule(ant_act_func=AND_min,
                  ants=[(car_speed, 'fast'), (car_acc, 'rising')],
                  cons=[(car_pedal, 'release')], impl_func=MIN),

        FuzzyRule(ant_act_func=AND_min,
                  ants=[(car_speed, 'slow'), (car_acc, 'constant')],
                  cons=[(car_pedal, 'push')], impl_func=MIN),

        FuzzyRule(ant_act_func=AND_min,
                  ants=[(car_speed, 'slow'), (car_acc, 'slowing')],
                  cons=[(car_pedal, 'push')], impl_func=MIN),

        FuzzyRule(ant_act_func=AND_min,
                  ants=[(car_speed, 'fast'), (car_acc, 'slowing')],
                  cons=[(car_pedal, 'nothing')], impl_func=MIN),

        FuzzyRule(ant_act_func=AND_min,
                  ants=[(car_speed, 'ok'), (car_acc, 'slowing')],
                  cons=[(car_pedal, 'nothing')], impl_func=MIN),

        FuzzyRule(ant_act_func=AND_min,
                  ants=[(car_speed, 'ok'), (car_acc, 'constant')],
                  cons=[(car_pedal, 'nothing')], impl_func=MIN),

        FuzzyRule(ant_act_func=AND_min,
                  ants=[(car_speed, 'ok'), (car_acc, 'rising')],
                  cons=[(car_pedal, 'nothing')], impl_func=MIN),

        FuzzyRule(ant_act_func=AND_min,
                  ants=[(car_speed, 'slow'), (car_acc, 'rising')],
                  cons=[(car_pedal, 'nothing')], impl_func=MIN)
    ]

    fis = FIS(rules=car_rules, aggr_func=np.max,
              defuzz_func=COA_func)

    input_values = {'speed': 10, 'acceleration': -1}
    predicted_value = fis.predict(input_values)["pedal"]

    expected_value = 38.5539690634
    print("predicted value: {}".format(predicted_value))
    print("expected  value: {}".format(expected_value))
    print("difference     : {}".format(expected_value - predicted_value))


if __name__ == '__main__':
    car_accel_problem()
