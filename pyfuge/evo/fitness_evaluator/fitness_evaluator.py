from abc import ABCMeta


class FitnessEvaluator(metaclass=ABCMeta):
    def eval(self, y_pred, y_true):
        """

        :param y_pred: a vector (or matrix if multiple consequents) containing
        the predicted outputs given the train dataset
        :param dataset:
        :return: a fitness float value for this fuzzy system
        """
        raise NotImplementedError()
