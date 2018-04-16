import numpy as np

from evo.dataset.pf_dataset import PFDataset
from evo.experiment.base.experiment import Experiment
from evo.fitness_evaluator.fitness_evaluator import FitnessEvaluator
from evo.helpers.ind_2_ifs import Ind2IFS


class SimpleEAExperiment(Experiment):
    """
    A class that performs an experiment using a simple EA (evolutionary
    algorithm) with DEAP library.
    """

    # @profile(sort="tottime", filename="/tmp/yolo.profile")
    def __init__(self, dataset: PFDataset, ind2ifs: Ind2IFS,
                 fitevaluator: FitnessEvaluator, **kwargs):
        super(SimpleEAExperiment, self).__init__(dataset, ind2ifs, fitevaluator,
                                                 **kwargs)

        import random
        from deap import creator, base, tools, algorithms

        target_length = self._ind2ifs.ind_length()

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

        def eval_ind(ind):
            # ifs = self._ind2ifs.convert(ind)
            # fitness = self._fiteval.eval(ifs, self._dataset)

            y_preds = self._ind2ifs.predict(ind)
            fitness = self._fiteval.eval_fitness(y_preds, self._dataset)
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
                            halloffame=hof,
                            stats=stats)
        self._top_n = tools.selBest(population, k=3)

    def get_top_n(self):
        return self._top_n

        # print("top_n")
        # for tn in top_n:
        #     print(tn)
