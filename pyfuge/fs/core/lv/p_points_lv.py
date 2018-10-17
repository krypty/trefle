from pyfuge.fs.core.lv.linguistic_variable import LinguisticVariable
from pyfuge.fs.core.mf.lin_piece_wise_mf import LinPWMF
from pyfuge.fs.view.lv_viewer import LinguisticVariableViewer


def generate_labels(n_points):
    def get_labels(n, prefix, labels=None):
        if labels is None:
            _labels = []
        else:
            _labels = labels

        # base recursion case
        if n == -1:
            return _labels

        # other cases
        if n > 2:
            new_label = "{} very {}".format(n, prefix)
        else:
            new_label = n * "very " + prefix

        _labels.append(new_label)

        return get_labels(n - 1, prefix, _labels)

    k = n_points - 1  # minus one for the "medium" label
    k1 = k // 2
    k = k - k1
    return [*get_labels(k - 1, "low"), "medium", *reversed(get_labels(k1 - 1, "high"))]


class PPointsLV(LinguisticVariable):
    """
    Syntactic sugar for simplified linguistic variable with N points (p1,
    p2, p3,...pN) and fixed labels ("very low", "low", "medium",
    "high", "very ...").


      ^
      | low      medium           high
    1 |XXXXX       X          XXXXXXXXXXXX
      |     X     X  X       XX
      |      X   X    X    XX
      |       X X      XX X
      |       XX        XXX
      |      X  X     XX   XX
      |     X    X XX       XX
      |    X       X          XX
    0 +-------------------------------------->
           p1     p2          p3


    """

    def __init__(self, name, p_points):
        if not len(p_points) > 1:
            raise ValueError("there must be at least 2 points")

        if p_points != sorted(p_points):
            raise ValueError("p_points must be increasing values")

        if len(p_points) == 2:
            super(PPointsLV, self).__init__(
                name,
                ling_values_dict={
                    "low": LinPWMF([p_points[0], 1], [p_points[1], 0]),
                    "high": LinPWMF([p_points[0], 0], [p_points[1], 1]),
                },
            )
        else:
            labels = generate_labels(len(p_points))
            mfs = self._create_mfs(p_points)

            ling_values_dict = {label: lv for label, lv in zip(labels, mfs)}
            super(PPointsLV, self).__init__(name, ling_values_dict)

    @staticmethod
    def _create_mfs(p_points):
        mf_values = len(p_points) * [0]

        for i in range(len(p_points)):
            p_args = [[j, k] for j, k in zip(p_points, mf_values)]
            p_args[i][1] = 1
            yield LinPWMF(*p_args)


if __name__ == "__main__":

    name = "yolo"
    a = [1.1, 4.4, 6.6, 7.7, 8.8]
    lv = PPointsLV(name, a)
    LinguisticVariableViewer(lv).show()
