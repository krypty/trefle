from core.membership_functions.lin_piece_wise_mf import LinPWMF


class TriangularMF(LinPWMF):
    """
    Assumptions:
    - mf values are bound to [0, 1]
    """

    def __init__(self, p_min, p_mid, p_max, n_points=50):
        super().__init__([p_min, 0], [p_mid, 1], [p_max, 0], n_points=n_points)


if __name__ == '__main__':
    mf = TriangularMF(p_min=2, p_mid=15, p_max=16, n_points=50)
    print(mf.fuzzify(2.5))
    assert mf.fuzzify(-2) == 0

    # large margin because it is rounded (i.e. limited by the number of points)
    assert abs(mf.fuzzify(2.5) - 0.03846) < 0.1

    assert mf.fuzzify(15) == 1
    assert mf.fuzzify(118) == 0
