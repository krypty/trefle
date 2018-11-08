from abc import ABCMeta


class FISIndividual(metaclass=ABCMeta):
    """
    This class offer two functions:
    * convert(): convert an individual (i.e. a list of float) into a FIS object.
    This latter will allow the use of the visualization tools from fuzzy_systems
    * predict(): using an individual (i.e. still the list of float) build a FIS
    (no matter how, e.g. using py_fiseval or (Singleton)FIS), execute this FIS
    on all the dataset and return the y_pred
    """

    def __init__(self):
        self._ind_len = None

    def ind_length(self):
        return self._ind_len

    def predict(self, ind):
        """
        Given an individual returns the y_pred for a given dataset.
        :param ind: an individual
        :return: y_pred
        """
        raise NotImplementedError()


class Clonable(metaclass=ABCMeta):
    """
    If the individuals of a FISIndividual subclass are not immutable then the
    class must provide a way to deep copy/clone the individual
    """

    @staticmethod
    def clone(ind):
        """
        for example:
            ind_copy = ind.copy()
            return ind_copy
        """
        raise NotImplementedError()
