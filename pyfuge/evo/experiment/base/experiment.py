from abc import ABCMeta, abstractmethod
from typing import Callable


class Experiment(metaclass=ABCMeta):
    def __init__(self, fitness_func: Callable, **kwargs):
        """

        :param fitness_func: a function that will return a fitness scalar.
        :param kwargs: other experiment parameters
        """
        self._fitness_func = fitness_func
        self._kwargs = kwargs

    def _post_init(self):
        self._warn_unused_args()

    def _warn_unused_args(self):
        if len(self._kwargs) > 0:
            cls_name = Experiment.__name__
            unused_args = ", ".join(self._kwargs.keys())
            print(
                "[{}] warning: the following arguments have been "
                "ignored: {}".format(cls_name, unused_args)
            )

    @abstractmethod
    def get_logbook(self):
        raise NotImplementedError()

    @abstractmethod
    def run(self):
        raise NotImplementedError()
