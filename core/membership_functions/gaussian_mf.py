import numpy as np

from core.membership_functions.free_shape_mf import FreeShapeMF


class GaussianMF(FreeShapeMF):
    """
    This represents a Gaussian MF. It may not be useful
    but it is implemented for learning purpose and especially
    for the following tip:

    If a membership can be expressed as a (simple) function, then overriding
    self.fuzzify() can be more efficient than computing a bunch of points a
    priori.
    /!\ Be careful to not use this kind of trick if you want to use it as a
    consequent
    """

    def __init__(self, mu, sigma):
        in_values = []
        mf_values = []
        self.__mu = mu
        self.__v = sigma ** 2
        super().__init__(in_values, mf_values)

    def fuzzify(self, in_value: float):
        # gaussian equation,
        # source: https://en.wikipedia.org/wiki/Normal_distribution
        return (1.0 / (np.sqrt(2.0 * np.pi * self.__v))) * np.exp(
            -((in_value - self.__mu) ** 2) / (2 * self.__v))


if __name__ == '__main__':
    mf = GaussianMF(mu=3, sigma=4)
    print(mf.fuzzify(0))
    print(mf.fuzzify(3))
    print(mf.fuzzify(10))
    print(mf.fuzzify(1000))
