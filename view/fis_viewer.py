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

        # axarr[0, 2].text(0, 0, "jdslkajdska")
        # axarr[1, 2].text(0, 0, "yolo")
        # axarr[0, 2].get_shared_x_axes().join(axarr[0, 2], axarr[1, 2])
        # axarr[1, 2].get_shared_x_axes().join(axarr[1, 2], axarr[2, 2])

        [ax.axis("off") for ax in axarr.flat]

        for line, r in enumerate(self.__fis.rules):
            self._create_rule_plot(r, ax_line=axarr[line, :],
                                   max_ants=max_ants, max_cons=max_cons)

        # all columns of consequents share the same x axe per column
        ax_cons_cols = axarr[:, -max_cons:]

        print("axarr", axarr.shape)

        print(ax_cons_cols)

        # j = 0
        # for ax_col in ax_cons_cols:
        #     j += 1
        #     for i in range(1, len(ax_col)):
        #         ax_col[i].text(4, 0, str(i) + ", " + str(j))
        #         ax_col[i - 1].get_shared_x_axes().join(ax_col[i - 1], ax_col[i])

        self._plot_rows_cols_labels(axarr, max_ants, max_cons)

        self._plot_aggregation(axarr[-1, -1])

    def show(self):
        plt.tight_layout()
        plt.show()

    def _create_rule_plot(self, r: FuzzyRule, ax_line, max_ants, max_cons):
        n_rule_members = len(ax_line)

        self._plot_ants(ax_line[:max_ants], r.antecedents, n_rule_members)
        self._plot_cons(ax_line[-max_cons:], r.consequents,
                        n_rule_members)

        # # share the x axe for all members of this rule
        # for i in range(1, len(ax_line)):
        #     ax_line[i - 1].get_shared_x_axes().join(ax_line[i - 1], ax_line[i])


        # for ant_or_cons, ax in zip_longest(ants_and_cons, ax_line,
        #                                    fillvalue=None):
        #     if ant_or_cons is None:
        #         ax.axis("off")
        #     else:
        #         mf = ant_or_cons[0][ant_or_cons[1]]
        #         label = "[{}] {}".format(ant_or_cons[0].name, ant_or_cons[1])
        #         MembershipFunctionViewer(mf, ax=ax, label=label)

    def _plot_ants(self, axarr, antecedents, n_rule_members):
        for ant, ax, i in zip_longest(antecedents, axarr,
                                      range(n_rule_members),
                                      fillvalue=None):

            if ant is None:
                continue

            ax.axis("on")
            ax.set_facecolor(ANTECEDENTS_BACKGROUND_COLOR)
            mf = ant[0][ant[1]]
            label = "[{}] {}".format(ant[0].name, ant[1])
            MembershipFunctionViewer(mf, ax=ax, label=label)

    def _plot_cons(self, axarr, consequents, n_rule_members):
        # assumption: each rule has the same number and names of consequents
        sorted_consequents = sorted(consequents, key=lambda c: c[0].name)

        for cons, ax, i in zip_longest(sorted_consequents, axarr,
                                       range(n_rule_members),
                                       fillvalue=None):
            if cons is None:
                continue

            ax.axis("on")
            ax.set_facecolor(CONSEQUENTS_BACKGROUND_COLOR)
            mf = cons[0][cons[1]]
            label = "[{}] {}".format(cons[0].name, cons[1])
            MembershipFunctionViewer(mf, ax=ax, label=label)

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

    def _plot_aggregation(self, ax):
        ax.axis("on")
        ax.text(0, 0.5, "aggregation will take place here")
