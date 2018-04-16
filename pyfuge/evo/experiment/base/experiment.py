from evo.dataset.pf_dataset import PFDataset
from evo.fitness_evaluator.fitness_evaluator import FitnessEvaluator
from evo.helpers.ind_2_ifs import Ind2IFS


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
