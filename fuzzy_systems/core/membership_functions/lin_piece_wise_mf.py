import numpy as np

from fuzzy_systems.core.membership_functions.free_shape_mf import FreeShapeMF


def gen_line(p0, p1, n_points):
    points_same_x = abs(p1[0] - p0[0]) < 1e-6
    points_same_y = abs(p1[1] - p0[1]) < 1e-6
    if points_same_y and points_same_x:
        return [p0[0]], [p0[1]]

    if points_same_x:
        return [p0[0], p1[0]], [p0[1], p1[1]]

    xs = np.linspace(p0[0], p1[0], n_points)
    slope = (p1[1] - p0[1]) / (p1[0] - p0[0])
    ys = [slope * (x - p0[0]) + p0[1] for x in xs]

    return xs, ys


class LinPWMF(FreeShapeMF):
    """
    This class produce a "linear piece-wise function"-like membership function
    This is a more "generic" class to produce linear, triangular and
    trapezoidal MF.

    Feel free to derive this class to create a TriangularMF, LinearMF,...
    """

    def __init__(self, *p_args, n_points=50):
        """
        Create a "linear piece-wise function"-like membership function
        :param p_args: this should at least contain 2 items. Each item is
        a point represented as an array [x, y] i.e. a point of the desired mf
        f(x) = y where x is a crisp value and y the assign mf value which is
        bound to [0,1]
        :param n_points: as only two points are needed to draw a line, this
        parameter let you choose the granularity (i.e. the number of points)
        to compute between all the piece-wise elements of the function.
        """
        assert len(p_args) >= 2, "Cannot produce MF with less than 2 points"
        n_pts = n_points
        n_pts_part = int(1 / len(p_args) * n_pts)

        in_values = []
        mf_values = []

        for i in range(1, len(p_args)):
            xs, ys = gen_line(p_args[i - 1], p_args[i], n_points=n_pts_part)
            # n_pts -= n_pts_part

            in_values.extend(xs)
            mf_values.extend(ys)

        super(LinPWMF, self).__init__(in_values, mf_values)


if __name__ == '__main__':
    from view.mf_viewer import MembershipFunctionViewer

    lin1 = LinPWMF([0, 0], [2, 1], [5, 1], [6, 0.5], [10, 0])
    MembershipFunctionViewer(lin1).show()
    x = abs(lin1.fuzzify(-133) - 0)
    assert x < 1e-2, x

    x = abs(lin1.fuzzify(1) - 0.50)
    assert x < 1e-2, x

    x = abs(lin1.fuzzify(3) - 1.0)
    assert x < 1e-2, x

    x = abs(lin1.fuzzify(6) - 0.5)
    assert x < 1e-2, x

    x = abs(lin1.fuzzify(8) - 0.25)
    assert x < 1e-2, x

    x = abs(lin1.fuzzify(120) - 0)
    assert x < 1e-2, x
