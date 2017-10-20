import numpy as np

from core.membership_functions.free_shape_mf import FreeShapeMF
from view.mf_viewer import MembershipFunctionViewer


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
        v = sigma ** 2

        # in order to get 99% of the input space,
        # see https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule
        in_values = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 25)
        mf_values = [self.gaussian(i, mu, v) for i in in_values]

        # scale the gaussian to y_max = 1 because Fuzzy sets should be in [0, 1]
        mf_values_max = max(mf_values)
        mf_values = [val / mf_values_max for val in mf_values]

        super().__init__(in_values, mf_values)

    @staticmethod
    def gaussian(x: float, mu: float, v: float):
        """
        Gaussian equation
        source: https://en.wikipedia.org/wiki/Normal_distribution
        :param x: value to compute
        :param mu: mean
        :param v: variance i.e. sigma*sigma
        :return: gaussian value of x
        """
        return (1.0 / (np.sqrt(2.0 * np.pi * v))) * np.exp(
            -((x - mu) ** 2) / (2 * v))


if __name__ == '__main__':
    mf = GaussianMF(mu=30, sigma=4)
    MembershipFunctionViewer(mf).show()
    print(mf.fuzzify(0))
    print(mf.fuzzify(3))
    print(mf.fuzzify(10))
    print(mf.fuzzify(1000))
