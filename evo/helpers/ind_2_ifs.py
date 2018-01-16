from abc import ABCMeta


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
