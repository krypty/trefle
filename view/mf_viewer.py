from core.membership_functions.free_shape_mf import FreeShapeMF
from core.membership_functions.lin_piece_wise_mf import LinPWMF
from view.viewer import Viewer


class MembershipFunctionViewer(Viewer):
    def __init__(self, mf: FreeShapeMF, label="", ax=None):
        super(MembershipFunctionViewer, self).__init__(ax)
        self.__mf = mf
        self.__label = label

        self.get_plot(self._ax)

    def fuzzify(self, in_value):
        fuzzified = self.__mf.fuzzify(in_value)

        self._ax.plot([in_value], [fuzzified], 'ro')
        self._ax.plot([in_value, in_value], [0, fuzzified], 'r')

    def get_plot(self, ax):
        ax.scatter(self.__mf.in_values, self.__mf.mf_values,
                   s=10, label=self.__label, c="green")
        ax.plot(self.__mf.in_values, self.__mf.mf_values,
                c="green")
        # ax.set_ylim([-0.1, 1.1])
        ax.legend()
        return ax


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots()

    mf = LinPWMF([17, 0], [20, 1], [26, 1], [29, 0])
    mfv = MembershipFunctionViewer(mf, ax=ax, label="N/A")
    mfv.fuzzify(22.5)

    plt.legend()
    plt.show()
