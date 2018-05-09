from sklearn.metrics import mean_absolute_error

from pyfuge.evo.fitness_evaluator.fitness_evaluator import FitnessEvaluator


class PyFUGEFitnessEvaluator(FitnessEvaluator):

    @staticmethod
    def _compute_metric(y_pred, y_true):
        # return -((y_pred - y_true) ** 2).sum(axis=None)

        return 1 - mean_absolute_error(y_true, y_pred)
        # return 1.0 - np.mean(np.mean((y_true - y_pred) ** 2, axis=0))

        # y_pred_bin = np.where(y_pred >= 0.5, 1, 0)
        #
        # n_good = 0
        # for row in range(y_pred.shape[0]):
        #     if np.all(np.equal(y_pred_bin[row], y_true[row])):
        #         n_good += 1
        # return n_good / float(y_pred.shape[0])

    def eval(self, y_preds, y_true):
        return self._compute_metric(y_preds, y_true)
