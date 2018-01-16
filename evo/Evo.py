from abc import ABCMeta

import numpy as np
from profilehooks import profile


def XXXDataset2PFDataset(orig_dataset):
    """
    Example of a function that convert a non-PyFUGE dataset to a PyFUGE dataset
    :param orig_dataset:
    :return: an instance of PF
    """
    return pf_dataset


class PFDataset:
    # add a version number in the case where serialization happen
    __dataset_version = 1

    def __init__(self, X, y, X_names=None, y_names=None):
        """
        PyFUGE-formatted dataset. You might create a class or method that
        convert your dataset into this format in order to use PyFUGE.

        :param X: features/variables as an np.ndarray (NxM) where
        N is the number of cases/observations and M is the number of variables
        per observation. Invalid values must be set as np.nan

        :param y: labels/classes for each observation represented as an
        np.ndarray (Nx1) where N is the number of observations. Each class must
        be represented by a unique number. TODO: should work great for numerical
        and categorical ordinal variables but not for categorical non-ordinal
        classes.

        :param X_names: name of the observations, if any. Might be useful to
        retrieve a particular observation afterwards
        :param y_names: name of the classes
        """
        self.X = X
        self.y = y
        self.X_names = X_names
        self.y_names = y_names

        self.N_OBS = X.shape[0]
        self.N_VARS = X.shape[1]
        self.N_CLASSES = np.unique(y).shape[0]

        # TODO make a Pandas DF ?


class IFS:
    def __init__(self):
        pass


class FitnessEvaluator(metaclass=ABCMeta):
    def eval(self, ifs: IFS, dataset: PFDataset):
        """
        evaluate the IFS against the dataset and return a fitness value for
        this IFS. The user of this class chooses the way the fitness is computed
        :param ifs:
        :param dataset:
        :return: a fitness float value for this IFS
        """
        raise NotImplementedError()

    def eval_fitness(self, y_pred, dataset: PFDataset):
        raise NotImplementedError()


class Ind2IFS(metaclass=ABCMeta):
    def __init__(self):
        self._ind_len = None

    def ind_length(self):
        return self._ind_len

    def convert(self, ind):
        """
        Convert an individual (evolution) to a IFS (FuzzySystem)
        :param ind: an individual
        :return: a IFS
        """
        raise NotImplementedError()

    def predict(self, ind):
        raise NotImplementedError()


# def ind2ifs(ind):
#     # TODO convert ind to ifs
#     ifs = None
#     return ifs


class Experiment:
    def __init__(self, dataset: PFDataset, ind2ifs: Ind2IFS,
                 fitevaluator: FitnessEvaluator, **kwargs):
        """

        :param dataset: a PyFUGE-formatted dataset
        :param ind2ifs: a function that converts an individual to an IFS
        (interpretable fuzzy system)
        :param fitevaluator: an instance of FitnessEvaluator that returns a
        fitness from a IFS
        :param kwargs: other experiment parameters
        """
        self._dataset = dataset
        self._ind2ifs = ind2ifs
        self._fiteval = fitevaluator

        self._kwargs = kwargs

    def run(self):
        pass


class SimpleEAExperiment(Experiment):
    """
    A class that performs an experiment using a simple EA (evolutionary
    algorithm) with DEAP library.
    """

    @profile(sort="tottime", filename="/tmp/yolo.profile")
    def __init__(self, dataset: PFDataset, ind2ifs: Ind2IFS,
                 fitevaluator: FitnessEvaluator, **kwargs):
        super(SimpleEAExperiment, self).__init__(dataset, ind2ifs, fitevaluator,
                                                 **kwargs)

        import random
        from deap import creator, base, tools, algorithms

        # N_RULES = self._kwargs.get("N_RULES") or 3
        # N_VARS = dataset.N_VARS
        # MAX_VAR_PER_RULE = self._kwargs.get(
        #     "MAX_VAR_PER_RULE") or N_VARS

        # n_p = N_VARS
        # n_d = N_VARS
        # n_a = N_VARS * N_RULES
        # n_c = N_RULES
        # target_length = n_p + n_d + n_a + n_c

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
        toolbox.register("select", tools.selTournament, tournsize=5)

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        N_POP = self._kwargs.get("N_POP") or 100
        population = toolbox.population(n=N_POP)

        NGEN = self._kwargs.get("N_GEN") or 10

        hof = tools.HallOfFame(self._kwargs.get("HOF") or 2)

        algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.1, ngen=NGEN,
                            halloffame=hof,
                            stats=stats)
        top_n = tools.selBest(population, k=3)

        print("top_n")
        for tn in top_n:
            print(tn)


class CoCoExperiment(Experiment):
    """
    A class that performs an experiment using a cooperative-coevolution
    algorithm with DEAP library.
    """

    def __init__(self):
        super(CoCoExperiment, self).__init__()


class CustomExperiment:
    """
    TODO user: create a totally custom class that perform an experiment using
    the fuzzy engine (core methods/classes).
    """
    pass
