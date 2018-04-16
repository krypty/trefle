from fuzzy_systems.core.membership_functions.lin_piece_wise_mf import LinPWMF


class TrapMF(LinPWMF):
    """
    Assumptions:
    - mf values are bound to [0, 1]

    This class is more an example of how you can derive LinPWMF
    """

    def __init__(self, p0, p1, p2, p3=None, n_points=50):
        """
        Create a trapezoidal mf if p0,p1,p2,p3 are given.
        Otherwise, if p3 is missing, create a triangular mf
        """
        args = [p0, 0], [p1, 1], [p2, 1], [p3, 0]

        if p3 is None:
            args = [p0, 0], [p1, 1], [p1, 1], [p2, 0]

        super().__init__(*args, n_points=n_points)
