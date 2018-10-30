from pyfuge.fs.core.lv.linguistic_variable import \
    LinguisticVariable
from pyfuge.fs.core.mf.lin_piece_wise_mf import \
    LinPWMF
from pyfuge.fs.view.mf_viewer import MembershipFunctionViewer
from pyfuge.fs.view.viewer import Viewer


class LinguisticVariableViewer(Viewer):
    def __init__(self, lv, ax=None):
        """

        :type lv: LinguisticVariable
        """
        super(LinguisticVariableViewer, self).__init__(ax)
        self.__lv = lv

        self._viewers = self.get_plot(self._ax)

    def fuzzify(self, in_value):
        [v.fuzzify(in_value) for v in self._viewers]
        return self

    def get_plot(self, ax):
        ax.set_title("MF: {}".format(self.__lv.name))
        ax.set_ylim([-0.1, 1.1])
        viewers = []
        for name in self.__lv.labels_name:
            mf = self.__lv[name]
            viewers.append(MembershipFunctionViewer(mf, label=name, ax=ax))
            # ax.set_title()
            # ax.scatter(mf.in_values, mf.mf_values, label=name)
        ax.legend(loc="best")
        return viewers


if __name__ == '__main__':
    from matplotlib import pyplot as plt

    fig, axs = plt.subplots(3, figsize=(12, 8))

    for ax in axs:
        lv_temp = LinguisticVariable(name="temperature", ling_values_dict={
            "cold": LinPWMF([17, 1], [20, 0]),
            "warm": LinPWMF([17, 0], [20, 1], [26, 1], [29, 0]),
            "hot": LinPWMF([26, 0], [29, 1])
        })
        viewer = LinguisticVariableViewer(lv_temp, ax=ax)
        viewer.fuzzify(26.6)
        viewer.fuzzify(21.8)

    fig.tight_layout()
    plt.show()

    # input("Please type ENTER")
