from abc import ABCMeta

from pyfuge.evo.dataset.pf_dataset import PFDataset


class FitnessEvaluator(metaclass=ABCMeta):
    def eval(self, ifs, dataset: PFDataset):
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
