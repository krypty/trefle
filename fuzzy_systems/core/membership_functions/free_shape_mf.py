import numpy as np


class FreeShapeMF:
    def __init__(self, in_values, mf_values):
        """
        Build a membership function by defining input values (x axis, in_values)
        and corresponding output values (y axis, mf_values)

        This class is the most basic way available to create membership
        functions. Other classes in the same package, such as LinPWMF, ease the
        build of well-known shaped membership functions.

        :param in_values:
        :param mf_values:
        """
        assert len(in_values) == len(
            mf_values), "Input and MF values are not the same length"

        self._in_values = np.array(in_values)
        self._mf_values = np.array(mf_values)

    def fuzzify(self, in_value: float):
        # return the nearest mf value for a given in_value using interpolation
        return np.interp(in_value, self._in_values, self._mf_values)

    @property
    def in_values(self):
        return self._in_values

    @property
    def mf_values(self):
        return self._mf_values


if __name__ == '__main__':
    in_values = [x for x in range(10)]
    mf = FreeShapeMF(
        in_values=in_values,
        mf_values=[x ** 2 for x in in_values]
    )

    # FIXME: this will now fail because this no more rounded but interpolated
    assert mf.fuzzify(1.6) == 4
    assert mf.fuzzify(3.2) == 9
    assert mf.fuzzify(-1.6) == 0
    assert mf.fuzzify(14566) == 81
