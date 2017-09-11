from abc import ABCMeta, abstractmethod


class Viewer(metaclass=ABCMeta):
    def __init__(self, ax):
        self._ax = ax

    @abstractmethod
    def get_plot(self, ax):
        raise NotImplementedError()
