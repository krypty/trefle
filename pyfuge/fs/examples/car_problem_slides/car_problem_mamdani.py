import numpy as np

from pyfuge.fs.core.fis.fis import FIS, COA_func, AND_min, MIN
from pyfuge.fs.core.linguistic_variables.linguistic_variable import \
    LinguisticVariable
from pyfuge.fs.core.membership_functions.lin_piece_wise_mf import \
    LinPWMF
from pyfuge.fs.core.rules.fuzzy_rule import FuzzyRule
from pyfuge.fs.core.rules.fuzzy_rule_element import Antecedent, \
    Consequent
from pyfuge.fs.view.fis_surface import show_surface


def car_accel_problem():
    car_speed = LinguisticVariable(name='speed', ling_values_dict={
        'slow': LinPWMF([-0.2, 1], [0, 0]),
        'ok': LinPWMF([-0.2, 0], [0, 1], [0.15, 0]),
        'fast': LinPWMF([0, 0], [0.15, 1], [1, 1], [2, 0])
    })

    car_acc = LinguisticVariable(name='speed_change', ling_values_dict={
        'slowing': LinPWMF([-0.3, 1], [0, 0]),
        'constant': LinPWMF([-0.3, 0], [0, 1], [0.3, 0]),
        'rising': LinPWMF([0, 0], [0.3, 1])
    })

    car_action = LinguisticVariable(name='action', ling_values_dict={
        'release': LinPWMF([-1, 1], [0, 0]),
        'nothing': LinPWMF([-1, 0], [0, 1], [1, 0]),
        'push': LinPWMF([0, 0], [1, 1])
    })

    car_rules = [
        FuzzyRule(ant_act_func=AND_min,
                  ants=[Antecedent(car_speed, 'fast'),
                        Antecedent(car_acc, 'constant')],
                  cons=[Consequent(car_action, 'release')], impl_func=MIN),

        FuzzyRule(ant_act_func=AND_min,
                  ants=[Antecedent(car_speed, 'fast'),
                        Antecedent(car_acc, 'rising')],
                  cons=[Consequent(car_action, 'release')], impl_func=MIN),

        FuzzyRule(ant_act_func=AND_min,
                  ants=[Antecedent(car_speed, 'slow'),
                        Antecedent(car_acc, 'constant')],
                  cons=[Consequent(car_action, 'push')], impl_func=MIN),

        FuzzyRule(ant_act_func=AND_min,
                  ants=[Antecedent(car_speed, 'slow'),
                        Antecedent(car_acc, 'slowing')],
                  cons=[Consequent(car_action, 'push')], impl_func=MIN),

        FuzzyRule(ant_act_func=AND_min,
                  ants=[Antecedent(car_speed, 'fast'),
                        Antecedent(car_acc, 'slowing')],
                  cons=[Consequent(car_action, 'nothing')], impl_func=MIN),

        FuzzyRule(ant_act_func=AND_min,
                  ants=[Antecedent(car_speed, 'ok'),
                        Antecedent(car_acc, 'slowing')],
                  cons=[Consequent(car_action, 'nothing')], impl_func=MIN),

        FuzzyRule(ant_act_func=AND_min,
                  ants=[Antecedent(car_speed, 'ok'),
                        Antecedent(car_acc, 'constant')],
                  cons=[Consequent(car_action, 'nothing')], impl_func=MIN),

        FuzzyRule(ant_act_func=AND_min,
                  ants=[Antecedent(car_speed, 'ok'),
                        Antecedent(car_acc, 'rising')],
                  cons=[Consequent(car_action, 'nothing')], impl_func=MIN),

        FuzzyRule(ant_act_func=AND_min,
                  ants=[Antecedent(car_speed, 'slow'),
                        Antecedent(car_acc, 'rising')],
                  cons=[Consequent(car_action, 'nothing')], impl_func=MIN)
    ]

    fis = FIS(rules=car_rules, aggr_func=np.max,
              defuzz_func=COA_func)

    input_values = {'speed': 1, 'speed_change': 0.22}
    fis.predict(input_values)

    # fisv = FISViewer(fis)
    # # # # fisv.save("/tmp/out.png")
    # fisv.show()

    return fis


if __name__ == '__main__':
    fis = car_accel_problem()
    show_surface(fis, x_label="speed", y_label="speed_change", z_label="action",
                 n_pts=15, x_range=(-1, 1), y_range=(-1, 1), z_range=(-1, 1))
