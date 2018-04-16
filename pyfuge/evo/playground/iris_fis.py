from functools import lru_cache
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pyfuge.fuzzy_systems.core.fis.fis import AND_min, MIN
from pyfuge.fuzzy_systems.core.fis.singleton_fis import SingletonFIS
from pyfuge.fuzzy_systems.core.linguistic_variables.linguistic_variable import \
    LinguisticVariable
from pyfuge.fuzzy_systems.core.linguistic_variables.two_points_lv import \
    TwoPointsPDLV
from pyfuge.fuzzy_systems.core.membership_functions.singleton_mf import \
    SingletonMF
from pyfuge.fuzzy_systems.core.rules.default_fuzzy_rule import DefaultFuzzyRule
from pyfuge.fuzzy_systems.core.rules.fuzzy_rule import FuzzyRule
from pyfuge.fuzzy_systems.core.rules.fuzzy_rule_element import Antecedent, \
    Consequent
from pyfuge.fuzzy_systems.view.fis_viewer import FISViewer


def main():
    fname = r"../../datasets/iris.data"
    iris_dataset = pd.read_csv(fname, sep=",",
                               names=["SL", "SW", "PL", "PW", "OUT"])

    X = iris_dataset[["SL", "SW", "PL", "PW"]].values

    # Iris-XX -> XX
    y = iris_dataset[["OUT"]].apply(axis=1, func=lambda x: x[0][5:]).values

    # print(X.shape)
    # print(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    float_ant_lookup = ["low", "dont_care", "dont_care", "high"]

    def float2ant(flt):
        """
        Each value is
        transformed y = f(x) = int(x*4 + 0.5), 0 = LOW, 1 and 2 = DCare and 3 = HIGH
        :param flt:
        :return:
        """
        return float_ant_lookup[int(flt * 3 + 0.5)]

    def _create_rule(ling_vars, ants, cons):
        """
        Create rule
        :param ants: array of floats representing a0,a1,..aN.
        :param cons: >= 0.5 -> the consequent is "versicolor"
        :return:
        """

        # TODO: generalize to N consequents. e.g.
        # consequents = np.zeros()
        # consequents[roulette_idx] = 1 # roulette_idx = float2~ant~conseq(evo_cons_float)

        if cons >= 0.5:
            cons = [
                Consequent(lv_setosa, 0),
                Consequent(lv_versicolor, 1),
                Consequent(lv_virginica, 0),
            ]
        else:
            cons = [
                Consequent(lv_setosa, 0),
                Consequent(lv_versicolor, 0),
                Consequent(lv_virginica, 1),
            ]

        ants = [Antecedent(ling_vars[i], float2ant(a_i)) for i, a_i in
                enumerate(ants)]
        ants = [a for a in ants if not a.lv_value == "dont_care"]
        # print([a.lv_value for a in ants])

        # all antecedents have been set to dont_care
        if len(ants) == 0:
            return None

        r = FuzzyRule(
            ants=ants,
            ant_act_func=AND_min,
            cons=cons,
            impl_func=MIN
        )
        return r

    yes_no = {
        0: SingletonMF(0),
        1: SingletonMF(1)
    }

    lv_setosa = LinguisticVariable(name="setosa", ling_values_dict=yes_no)
    lv_virginica = LinguisticVariable(name="virginica", ling_values_dict=yes_no)
    lv_versicolor = LinguisticVariable(name="versicolor",
                                       ling_values_dict=yes_no)

    def _create_default_rule(out_set_ver_sir: List[int]):
        dr = DefaultFuzzyRule(
            cons=[
                Consequent(lv_setosa, out_set_ver_sir[0]),
                Consequent(lv_versicolor, out_set_ver_sir[1]),
                Consequent(lv_virginica, out_set_ver_sir[2]),
            ],
            impl_func=MIN
        )
        return dr

    # sl_min, sl_max =

    N_VARS = 4  # FIXME
    N_RULES = 3

    # def range_factory(key):
    #     col = iris_dataset.columns[key]
    #     return iris_dataset[col].min(), iris_dataset[col].max()
    #
    # class CachedDict(defaultdict):
    #     def __init__(self, missing_func):
    #         super(CachedDict, self).__init__()
    #         self._missing_func = missing_func
    #
    #     def __missing__(self, key):
    #         ret = self._missing_func(key)  # calculate default value
    #         self[key] = ret
    #         return ret
    #
    # var_ranges = CachedDict(missing_func=range_factory)

    @lru_cache(maxsize=None)
    def get_var_range(i):
        """
        Return min and max values for variable i
        :param i: variable i
        :return: (min_var, max_var)
        """
        # TODO: add cache with dict
        col = iris_dataset.columns[i]
        return iris_dataset[col].min(), iris_dataset[col].max()
        # return var_ranges[i]

    def _create_ling_var(i, p, d):
        min_v, max_v = get_var_range(i)
        real_p = min_v + p * (max_v - min_v)
        real_d = real_p + d * (max_v - real_p)

        return TwoPointsPDLV(name=iris_dataset.columns[i], p=real_p, d=real_d)

    # @profile(sort="tottime")
    def _create_fis(individual):
        """

        :param individual: list of floats of size N_VARS + N_VARS + N_VARS * N_RULES
        to represent respectively P, d and A parameters such as described in Table
        2.2 of the book
        :return:
        """
        # Build FIS from Fig 3.9 of Carlos Peña's book

        p_params = individual[:N_VARS]
        d_params = individual[N_VARS: N_VARS * 2]
        ants_params = individual[N_VARS * 2:-N_RULES]
        cons_params = individual[-N_RULES:]

        # print(individual)
        # print(p_params, d_params, ants_params, cons_params)

        it_ling_vars = enumerate(zip(p_params, d_params))
        ling_vars = [_create_ling_var(i, p, d) for i, (p, d) in it_ling_vars]

        # print("cons", cons_params)

        rules = [_create_rule(ling_vars, ants, cons) for ants, cons in
                 zip(np.array_split(ants_params, N_RULES), cons_params)]

        rules = [r for r in rules if r is not None]

        out_set_ver_sir = [1, 0, 0]
        dr = _create_default_rule(out_set_ver_sir)

        fis = SingletonFIS(
            rules=rules,
            default_rule=dr,
        )

        return fis

    def predict(fis, X, y):
        # fis.describe()
        y_preds = []
        for i, x in enumerate(X):
            out = fis.predict(
                {iris_dataset.columns[i]: xi for i, xi in enumerate(x)}
            )
            y_preds.append(out[y[i]])

        return y_preds

    def compute_metric(y_pred, vars_per_rule):
        y_true = np.ones_like(y_pred)
        error = -((y_pred - y_true) ** 2).mean(axis=None)

        return error + (-0.001 * vars_per_rule)

    # @profile(sort="tottime")
    def evaluate_ind(ind):
        fis = _create_fis(ind)
        vars_per_rule = sum([len(r.antecedents) for r in fis.rules])
        y_pred = predict(fis, X=X_train, y=y_train)
        metric = compute_metric(y_pred, vars_per_rule)
        return [metric]

    import random
    from deap import creator, base, tools, algorithms

    n_p = N_VARS
    n_d = N_VARS
    n_a = N_VARS * N_RULES
    n_c = N_RULES
    target_length = n_p + n_d + n_a + n_c

    print("target len", target_length)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    # import multiprocessing
    #
    # pool = multiprocessing.Pool()
    # toolbox.register("map", pool.map)

    toolbox.register("attr_bool", random.random)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_bool, n=target_length)
    toolbox.register("population", tools.initRepeat, list,
                     toolbox.individual)

    toolbox.register("evaluate", evaluate_ind)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutShuffleIndexes,
                     indpb=1.0 / target_length)
    toolbox.register("select", tools.selTournament, tournsize=5)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    population = toolbox.population(n=300)

    NGEN = 5

    hof = tools.HallOfFame(2)

    """
    HOF == kind of elitism : "[...] use of a HallOfFame in order to keep track of the best individual to appear in the evolution (it keeps it even in the case it extinguishes),"
    HOF: max = 1 if any of the fitness tuple is equal to 1
    example:
        - (0.5, 1.0, 0.33) --> 1.0
        - np.mean((0.5, 1.0, 0.33)) --> 0.XX --> 0.XX
        
    ERRATA: to use a fitness with multiple values, use:
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
    """

    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.1, ngen=NGEN,
                        # halloffame=hof,
                        stats=stats)
    topN = tools.selBest(population, k=3)

    print("len pop", len(population))
    print(get_var_range.cache_info())

    def view_top_n_fis(iris_dataset, topN):
        for t in topN:
            fis = _create_fis(t)
            fis.describe()
            fis.predict(
                {iris_dataset.columns[i]: xi for i, xi in enumerate([2.5] * 4)})
            print("fitness", t.fitness)
            fisv = FISViewer(fis)
            axarr = fisv.get_axarr()

            classes = np.unique(iris_dataset["OUT"].values)

            for r_i, r in enumerate(fis.rules):
                for a_i, a in enumerate(r.antecedents):
                    lv_name = a.lv_name.name

                    xs_classes = [
                        iris_dataset[iris_dataset["OUT"] == c][lv_name].values
                        for c
                        in classes]
                    # xs2 = iris_dataset[[r0a0_lv_name]].values[50:100]
                    # xs3 = iris_dataset[[r0a0_lv_name]].values[100:]

                    ys = np.zeros_like(xs_classes[0])

                    [axarr[r_i, a_i].scatter(xs, ys, s=2.0, alpha=0.1) for xs in
                     xs_classes]
                    # axarr[0, 0].scatter(xs, ys, s=1.0, alpha=0.1)
                    # axarr[0, 0].scatter(xs2, ys, s=1.0, alpha=0.1)
                    # axarr[0, 0].scatter(xs3, ys, s=1.0, alpha=0.1)
            fisv.show()
            print("---------")
            print("---------")

    # view_top_n_fis(iris_dataset, topN)
    #
    # # Test best FIS
    # for idx, ind in enumerate(topN):
    #     fis = _create_fis(ind)
    #
    #     pred_test = predict(fis, X, y)
    #
    #     n_good_preds = len([p for p in pred_test if p >= 0.5])
    #
    #     print("[{}] good preds {}/{} ({:.3f})".format(
    #         idx, n_good_preds, len(y), n_good_preds / float(len(y))))


if __name__ == '__main__':
    from time import time

    # from datetime import datetime
    #
    # print(str(datetime.now()))
    t0 = time()
    tick = lambda: print((time() - t0) * 1000)
    #
    main()
    tick()
    # print(str(datetime.now()))

    # main()
