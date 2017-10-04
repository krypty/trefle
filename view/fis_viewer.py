from itertools import zip_longest

from matplotlib import pyplot as plt

from core.fis.fis import FIS
from core.rules.fuzzy_rule import FuzzyRule
from view.mf_viewer import MembershipFunctionViewer

ANTECEDENTS_BACKGROUND_COLOR = (0.95, 0.95, 0.95)
CONSEQUENTS_BACKGROUND_COLOR = "white"


class FISViewer:
    def __init__(self, fis: FIS):
        self.__fis = fis

        n_rules = len(self.__fis.rules)
        max_ants = max([len(r.antecedents) for r in self.__fis.rules])
        max_cons = max([len(r.consequents) for r in self.__fis.rules])

        max_sum_ants_cons = max_ants + max_cons
        fig, axarr = plt.subplots(ncols=max_sum_ants_cons,
                                  nrows=n_rules + 1,
                                  figsize=(16, 9))

        # -1 because last line is for aggregation
        for row in range(axarr.shape[0] - 1):
            for col in range(max_ants, axarr.shape[1]):
                a = axarr[row - 1, col]
                b = axarr[row, col]
                a.get_shared_x_axes().join(a, b)
                a.get_shared_y_axes().join(a, b)

        # axarr[0, 2].text(0, 0, "jdslkajdska")
        # axarr[1, 2].text(0, 0, "yolo")
        # axarr[0, 2].get_shared_x_axes().join(axarr[0, 2], axarr[1, 2])
        # axarr[1, 2].get_shared_x_axes().join(axarr[1, 2], axarr[2, 2])

        [ax.axis("off") for ax in axarr.flat]

        for line, r in enumerate(self.__fis.rules):
            self._create_rule_plot(r, ax_line=axarr[line, :],
                                   max_ants=max_ants, max_cons=max_cons,
                                   rule_index=line)

        # all columns of consequents share the same x axe per column
        ax_cons_cols = axarr[:, -max_cons:]

        # j = 0
        # for ax_col in ax_cons_cols:
        #     j += 1
        #     for i in range(1, len(ax_col)):
        #         ax_col[i].text(4, 0, str(i) + ", " + str(j))
        #         ax_col[i - 1].get_shared_x_axes().join(ax_col[i - 1], ax_col[i])

        self._plot_rows_cols_labels(axarr, max_ants, max_cons)

        for cons_index, ax in enumerate(axarr[-1, max_ants:]):
            self._plot_aggregation(cons_index, ax)

    def show(self):
        plt.tight_layout()
        plt.show()

    def _create_rule_plot(self, r: FuzzyRule, ax_line, max_ants, max_cons,
                          rule_index):
        n_rule_members = len(ax_line)

        self._plot_ants(ax_line[:max_ants], r.antecedents, n_rule_members)
        self._plot_cons(ax_line[-max_cons:], r.consequents,
                        n_rule_members, rule_index)

    def _plot_ants(self, axarr, antecedents, n_rule_members):
        for ant, ax, i in zip_longest(antecedents, axarr,
                                      range(n_rule_members),
                                      fillvalue=None):

            if ant is None:
                continue

            ax.axis("on")
            ax.set_facecolor(ANTECEDENTS_BACKGROUND_COLOR)

            for mf in ant[0].ling_values.values():
                MembershipFunctionViewer(mf, ax=ax, color="gray", alpha=0.1)

            mf = ant[0][ant[1]]
            label = "[{}] {}".format(ant[0].name, ant[1])
            MembershipFunctionViewer(mf, ax=ax, label=label)

            # show last crisp inputs
            crisp_values = self.__fis.last_crisp_values
            in_value = crisp_values[ant[0].name]
            fuzzified = mf.fuzzify(in_value)

            ax.plot([in_value], [fuzzified], 'ro')
            ax.plot([in_value, in_value], [0, fuzzified], 'r')

    def _plot_cons(self, axarr, consequents, n_rule_members, rule_index):
        # assumption: each rule has the same number and names of consequents
        sorted_consequents = sorted(consequents, key=lambda c: c[0].name)

        for cons, ax, i in zip_longest(sorted_consequents, axarr,
                                       range(n_rule_members),
                                       fillvalue=None):
            # print(cons, ax, i)
            if cons is None:
                continue

            ax.axis("on")
            ax.set_facecolor(CONSEQUENTS_BACKGROUND_COLOR)
            mf = cons[0][cons[1]]
            label = "[{}] {}".format(cons[0].name, cons[1])
            MembershipFunctionViewer(mf, ax=ax, label=label)

            # print(self.__fis.last_implicated_consequents)
            mf_implicated = \
                self.__fis.last_implicated_consequents[cons[0].name][rule_index]
            MembershipFunctionViewer(mf_implicated, ax=ax,
                                     label=label + " implicated",
                                     color="orange")

    def _plot_rows_cols_labels(self, axarr, max_ants, max_cons):
        col_ants = ['Antecedent {}'.format(col + 1) for col in range(max_ants)]
        col_cons = ['Consequent {}'.format(col + 1) for col in range(max_cons)]
        rows = ['Rule {}'.format(row + 1) for row in range(axarr.shape[1] - 1)]

        for ax, col in zip(axarr[0], col_ants):
            ax.set_title(col)

        for ax, col in zip(axarr[0, max_ants:], col_cons):
            ax.set_title(col)

        for ax, row in zip(axarr[:, 0], rows):
            ax.set_ylabel(row, rotation=90, size='large')
            # ax.yaxis.set_label_coords(-0.15, 0.5)

    def _plot_aggregation(self, cons_index, ax):
        aggr_cons = self.__fis.last_aggregated_consequents

        cons_labels = list(aggr_cons.keys())
        mf = list(aggr_cons.values())[cons_index]
        print(mf)
        MembershipFunctionViewer(mf, ax=ax,
                                 label="[{}]".format(
                                     cons_labels[cons_index]) + " aggregated",
                                 color="orange")

        # show last crisp inputs
        crisp_values = self.__fis.last_crisp_values
        print(crisp_values)
        # in_value = crisp_values[ant[0].name]
        # fuzzified = mf.fuzzify(in_value)
        #
        # ax.plot([in_value], [fuzzified], 'ro')
        # ax.plot([in_value, in_value], [0, fuzzified], 'r')

        defuzz = list(self.__fis.last_defuzzified_outputs.values())[cons_index]
        ax.plot([defuzz, defuzz], [0, 1],
                label="output = {:.3f}".format(defuzz))
        ax.legend()

        ax.axis("on")
