from evo.dataset.pf_dataset import PFDataset
from evo.fitness_evaluator.fitness_evaluator import FitnessEvaluator


class PyFUGEFitnessEvaluator(FitnessEvaluator):

    @staticmethod
    def _compute_metric(y_pred, y_true):
        return -((y_pred - y_true) ** 2).mean(axis=None)

    def eval(self, ifs, dataset: PFDataset):
        pass

    def eval_fitness(self, y_preds, dataset: PFDataset):
        y_true = dataset.y
        return self._compute_metric(y_preds, y_true)
