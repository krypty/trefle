import random

import numpy as np
from deap import creator, base, tools, algorithms

from pyfuge.evo.dataset.pf_dataset import PFDataset
from pyfuge.evo.experiment.base.experiment import Experiment
from pyfuge.evo.fitness_evaluator.fitness_evaluator import FitnessEvaluator
from pyfuge.evo.helpers.fis_individual import FISIndividual


class SimpleEAExperiment(Experiment):
    """
    A class that performs an experiment using a simple EA (evolutionary
    algorithm) with DEAP library.
    """

    def __init__(self, dataset: PFDataset, fis_individual: FISIndividual,
                 fitevaluator: FitnessEvaluator, **kwargs):
        super(SimpleEAExperiment, self).__init__(dataset, fis_individual,
                                                 fitevaluator, **kwargs)

        target_length = self._fis_individual.ind_length()

        print("target len", target_length)

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)

        toolbox = base.Toolbox()

        toolbox.register("attr_bool", random.random)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.attr_bool, n=target_length)
        toolbox.register("population", tools.initRepeat, list,
                         toolbox.individual)

        def eval_ind(ind):
            y_preds = self._fis_individual.predict(ind)
            fitness = self._fiteval.eval(y_preds, self._dataset.y)
            return [fitness]

        toolbox.register("evaluate", eval_ind)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutShuffleIndexes,
                         indpb=1.0 / target_length)
        toolbox.register("select", tools.selTournament, tournsize=3)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)
        # stats.register("len", len)

        N_POP = self._kwargs.get("N_POP") or 100
        population = toolbox.population(n=N_POP)

        NGEN = self._kwargs.get("N_GEN") or 10

        hof = tools.HallOfFame(self._kwargs.get("HOF") or 5)

        algorithms.eaSimple(population, toolbox, cxpb=0.8, mutpb=0.3, ngen=NGEN,
                            halloffame=hof, stats=stats)
        self._top_n = tools.selBest(population, k=3)

    def get_top_n(self):
        return self._top_n
