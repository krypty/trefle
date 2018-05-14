from pyfuge.evo.dataset.pf_dataset import PFDataset
from pyfuge.evo.fitness_evaluator.fitness_evaluator import FitnessEvaluator
from pyfuge.evo.helpers.fis_individual import FISIndividual


class Experiment:
    def __init__(self, dataset: PFDataset, fis_individual: FISIndividual,
                 fitevaluator: FitnessEvaluator, **kwargs):
        """

        :param dataset: a PyFUGE-formatted dataset
        :param fis_individual: a class that uses an individual to return
        predictions
        :param fitevaluator: an instance of FitnessEvaluator that returns a
        fitness from predictions
        :param kwargs: other experiment parameters
        """
        self._dataset = dataset
        self._fis_individual = fis_individual
        self._fiteval = fitevaluator

        self._kwargs = kwargs

    def run(self):
        pass
