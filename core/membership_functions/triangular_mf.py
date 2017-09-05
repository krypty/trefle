import numpy as np

from core.membership_functions.free_shape_mf import FreeShapeMF


class TriangularMF(FreeShapeMF):
    """
    Assumptions:
    - mf values are bound to [0, 1]
    """

    def __init__(self, p_min, p_mid, p_max, n_points=10):
        n_pts_min_mid = n_points // 2
        n_pts_mid_max = n_points - n_pts_min_mid

        in_increasing = np.linspace(p_min, p_mid, n_pts_min_mid)
        in_decreasing = np.linspace(p_mid, p_max, n_pts_mid_max)
        in_values = np.append(in_increasing, in_decreasing)

        # increasing slope
        slope = 1 / (p_mid - p_min)
        out_increasing = [slope * (x - p_min) for x in in_increasing]

        # decreasing slope
        slope = -1 / (p_max - p_mid)
        out_decreasing = [slope * (x - p_max) for x in in_decreasing]

        mf_values = np.append(out_increasing, out_decreasing)

        # plt.scatter(in_values, mf_values)
        # plt.show()
        super().__init__(in_values, mf_values)


if __name__ == '__main__':
    mf = TriangularMF(p_min=2, p_mid=15, p_max=16, n_points=100)
    print(mf.fuzzify(2.5))
    assert mf.fuzzify(-2) == 0

    # large margin because it is rounded (i.e. limited by the number of points)
    assert abs(mf.fuzzify(2.5) - 0.03846) < 0.1
    
    assert mf.fuzzify(15) == 1
    assert mf.fuzzify(118) == 0
