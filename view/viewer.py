from matplotlib import pyplot as plt
from abc import ABCMeta, abstractmethod


class Viewer(metaclass=ABCMeta):
    def __init__(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        self._ax = ax

    @abstractmethod
    def get_plot(self, ax):
        raise NotImplementedError()


    def show(self):
        plt.show()
