from typing import Callable

from pyfuge.evo.dataset.pf_dataset import PFDataset
from pyfuge.evo.helpers.fis_individual import FISIndividual


class Experiment:
    def __init__(self, dataset: PFDataset, fis_individual: FISIndividual,
                 fitness_func: Callable, **kwargs):
        """

        :param dataset: a PyFUGE-formatted dataset
        :param fis_individual: a class that uses an individual to return
        predictions
        :param fitness_func: a function that will return a fitness scalar.
        :param kwargs: other experiment parameters
        """
        self._dataset = dataset
        self._fis_individual = fis_individual
        self._fitness_func = fitness_func

        self._kwargs = kwargs

    def run(self):
        pass
