import numpy as np


class FreeShapeMF:
    def __init__(self, in_values, mf_values):
        """
        TODO: todo...
        :param in_values:
        :param mf_values:
        """
        assert len(in_values) == len(
            mf_values), "Input and MF values are not the same length"

        assert all(
            [in_values[i - 1] <= in_values[i] for i in
             np.arange(1, len(in_values))]
        ), "Input values are not monotonic increasing"

        self.__in_values = np.array(in_values)
        self.__mf_values = np.array(mf_values)

    def fuzzify(self, in_value):
        # return the nearest mf value for a given in_value
        idx = (np.abs(self.__in_values - in_value)).argmin()
        return self.__mf_values[idx]


if __name__ == '__main__':
    in_values = [x for x in range(10)]
    mf = FreeShapeMF(
        in_values=in_values,
        mf_values=[x ** 2 for x in in_values]
    )

    assert mf.fuzzify(1.6) == 4
    assert mf.fuzzify(3.2) == 9
    assert mf.fuzzify(-1.6) == 0
    assert mf.fuzzify(14566) == 81
