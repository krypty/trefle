import math
import random
from typing import Callable

import numpy as np
from deap import creator, base, tools, algorithms

from trefle.evo.experiment.base.experiment import Experiment
from trefle.evo.helpers.fis_individual import FISIndividual


class SimpleEAExperiment(Experiment):
    """
    A class that performs an experiment using a simple EA (evolutionary
    algorithm) with DEAP library.
    """

    def run(self):
        algorithms.eaSimple(
            self._population, self._toolbox, cxpb=0.8,
            mutpb=0.3, ngen=self._NGEN,
            halloffame=self._hof, stats=self._stats,
            verbose=self._verbose)

        self._top_n = tools.selBest(self._population, k=3)

    def __init__(self, fis_individual: FISIndividual,
                 fitness_func: Callable, **kwargs):
        super(SimpleEAExperiment, self).__init__(fitness_func, **kwargs)

        self._fis_individual = fis_individual
        target_length = self._fis_individual.ind_length()

        try:
            # Don't recreate creator classes if they already exist. See issue 41
            creator.FitnessMax
        except AttributeError:
            creator.create("FitnessMax", base.Fitness, weights=(1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMax)

        self._toolbox = base.Toolbox()

        self._toolbox.register("attr_bool", random.random)
        self._toolbox.register("individual", tools.initRepeat,
                               creator.Individual,
                               self._toolbox.attr_bool, n=target_length)
        self._toolbox.register("population", tools.initRepeat, list,
                               self._toolbox.individual)

        def eval_ind(ind):
            y_pred_one_hot = self._fis_individual.predict(ind)
            y_true = np.argmax(self._dataset.y, axis=1)

            # TODO: do not threshold/binarize y_pred. Instead let the fitness
            # handle that otherwise we regression metrics are useless (even in
            # a classification problem)
            if y_pred_one_hot.shape[0] > 1:
                y_pred = np.argmax(y_pred_one_hot, axis=1)
            else:
                y_pred = np.round(y_pred_one_hot)

            fitness = self._fitness_func(y_true, y_pred)
            return fitness,  # DEAP expects a tuple for fitnesses

        self._toolbox.register("evaluate", eval_ind)
        self._toolbox.register("mate", tools.cxTwoPoint)
        self._toolbox.register("mutate", tools.mutShuffleIndexes,
                               indpb=1.0 / (10.0 * math.ceil(
                                   math.log(target_length, 10)))),
        self._toolbox.register("select", tools.selTournament, tournsize=3)

        self._verbose = kwargs.pop("verbose", False)
        self._stats = self.setup_stats(self._verbose)

        N_POP = self._kwargs.pop("N_POP", 100)
        self._population = self._toolbox.population(n=N_POP)

        self._NGEN = self._kwargs.pop("N_GEN", 10)

        self._hof = tools.HallOfFame(self._kwargs.pop("HOF", 5))

        self._post_init()

    @staticmethod
    def setup_stats(verbose):
        if not verbose:
            return None

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        # stats.register("len", len)
        return stats

    def get_top_n(self):
        return self._top_n
