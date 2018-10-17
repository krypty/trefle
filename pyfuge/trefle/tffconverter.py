import json
import os
from pyfuge_c import TrefleFIS

from pyfuge.evo.helpers.fuzzy_labels_generator import generate_labels
from pyfuge.fs.core.fis.fis import MIN, AND_min
from pyfuge.fs.core.fis.singleton_fis import SingletonFIS
from pyfuge.fs.core.lv.linguistic_variable import LinguisticVariable
from pyfuge.fs.core.lv.p_points_lv import PPointsLV
from pyfuge.fs.core.mf.singleton_mf import SingletonMF
from pyfuge.fs.core.rules.default_fuzzy_rule import DefaultFuzzyRule
from pyfuge.fs.core.rules.fuzzy_rule import FuzzyRule
from pyfuge.fs.core.rules.fuzzy_rule_element import Antecedent, Consequent
from pyfuge.fs.view.fis_viewer import FISViewer


class TffJsonToSingletonFIS:
    SUPPORTED_VERSION = 1

    def __init__(self, tff_str):
        self._tff_str = tff_str
        self._jfis = json.loads(self._tff_str)

    def convert(self):
        self._ensure_version()

        labels = generate_labels(n_labels=self._jfis["n_labels"])

        cons_labels = []
        for n_labels, k_classes in zip(
            self._jfis["n_labels_per_cons"], self._jfis["n_classes_per_cons"]
        ):
            if k_classes == 0:
                cons_labels.append(generate_labels(n_labels))
            else:
                cons_labels.append(list(range(n_labels)))

        # cons_labels = [
        #     generate_labels(n) for n, k in zip(self._jfis["n_labels_per_cons"], self._jfis["n_classes_per_cons"] ) if k == 0 else range(n)
        # ]

        lvs = self._parse_lvs()

        # TODO scale/scale back cons
        cons_lvs = self._parse_cons_lvs()

        rules = self._parse_rules(lvs, labels, cons_lvs, cons_labels)

        # for r in rules:
        #     print(r)

        default_rule = self._parse_default_rule(cons_lvs, cons_labels)

        # for lv in cons_lvs:
        #     LinguisticVariableViewer(lv).show()

        # fr = FuzzyRule()
        # rules = []
        # default_rule = []
        #
        fis = SingletonFIS(rules, default_rule)
        fis.describe()
        FISViewer(fis).show()
        # print(jfis)

        return fis

        # TODO return SingletonFIS

    def _ensure_version(self):
        if self._jfis["version"] != self.SUPPORTED_VERSION:
            raise ValueError(
                "Unsupported tff version! Currently supported: {}".format(
                    self.SUPPORTED_VERSION
                )
            )

    def _parse_lvs(self):
        lvs = self._jfis["linguistic_variables"]
        return {name: PPointsLV(name, p_pos) for name, p_pos in lvs.items()}

    @staticmethod
    def _create_singleton_lv(name, p_points, labels):
        ling_values_dict = {
            label: SingletonMF(point) for point, label in zip(p_points, labels)
        }
        return LinguisticVariable(name, ling_values_dict)

    def _parse_cons_lvs(self):
        cons_lvs = self._jfis["n_labels_per_cons"]
        n_classes_per_cons = self._jfis["n_classes_per_cons"]
        cons = []

        for i, n_label in enumerate(cons_lvs):
            cons_range = self._jfis["cons_range"][i]
            # n_classes = 0 --> regression, so use fuzzy labels
            if n_classes_per_cons[i] == 0:
                p_points = self._scale_back_cons(n_label, cons_range)
                labels = generate_labels(n_label)
            else:
                p_points = range(n_label)
                labels = range(n_label)

            lv = self._create_singleton_lv("out{}".format(i), p_points, labels)
            cons.append(lv)

        return cons

    def _parse_rules(self, lvs, labels, lvs_cons, cons_labels):
        return [
            self._parse_rule(jrule, lvs, labels, lvs_cons, cons_labels)
            for jrule in self._jfis["rules"]
        ]

    def _parse_rule(self, jrule, lvs, labels, lvs_cons, cons_labels):
        ants = [self._parse_ant(jant, lvs, labels) for jant in jrule[0]]

        cons = [
            self._parse_con(jrule[1][i], lvs_cons[i], cons_labels[i])
            for i in range(len(cons_labels))
        ]

        return FuzzyRule(ants=ants, ant_act_func=AND_min, cons=cons, impl_func=MIN)

    @staticmethod
    def _parse_ant(jant, lvs, labels):
        return Antecedent(lvs[jant[0]], labels[jant[1]])

    @staticmethod
    def _parse_con(jcon, lv, cons_label):
        return Consequent(lv, cons_label[int(jcon)])

    def _parse_default_rule(self, lvs_cons, cons_labels):
        jdef_cons = self._jfis["default_rule"]
        cons = [
            self._parse_con(jdef_cons[i], lvs_cons[i], cons_labels[i])
            for i in range(len(jdef_cons))
        ]
        return DefaultFuzzyRule(cons, impl_func=MIN)

    @staticmethod
    def _scale_back_cons(n_label, cons_range):
        c_range = cons_range[1] - cons_range[0]
        return [
            c_range * (i / float(n_label)) + cons_range[0]
            for i in range(1, n_label + 1)
        ]


class TffConverter:
    @staticmethod
    def to_fis(tff_str: str):
        """

        :param tff_str: if tff_str is a valid file path, this latter
        will be read and parsed. Otherwise tff_str will be interpreted as
        a json str and parsed directly.
        :return: a SingletonFIS instance
        """
        return TffJsonToSingletonFIS(tff_str).convert()
        # raise NotImplementedError

    @staticmethod
    def to_trefle_fis(tff_str):
        """

        :param tff_str: if tff_str is a valid file path, this latter
        will be read and parsed. Otherwise tff_str will be interpreted as
        a json str and parsed directly.
        :return: a TrefleFIS instance
        """

        if os.path.exists(tff_str):
            return TrefleFIS.from_tff_file(tff_str)
        else:
            return TrefleFIS.from_tff(tff_str)


if __name__ == "__main__":
    # from pyfuge.trefle.tffconverter import TffConverter as tamere

    # obj = TrefleFIS(30)
    # obj.predict()
    #
    # obj1 = TrefleFIS.from_tff("sjdlkaj")
    # obj1.predict()
    #
    # obj2 = TrefleFIS.from_tff_file("sjdlkaj")
    # obj2.predict()

    tff_str = r"""
   {
  "cons_range": [
    [0.0, 1.0],
    [0.0, 3.0],
    [20.051916073376123, 119.84553112721537]
  ],
  "default_rule": [1.0, 2.0, 1.0],
  "linguistic_variables": {
    "0": [13.796806451612902, 14.478387096774192, 16.523129032258062, 25.38367741935484],
    "1": [25.925806451612903, 29.741290322580646, 32.60290322580645, 34.51064516129033],
    "14": [0.004559806451612903, 0.014998096774193549, 0.02164064516129032, 0.030181064516129036],
    "24": [0.08582451612903226, 0.09070935483870968, 0.12978806451612904, 0.13955774193548387],
    "25": [0.11491032258064515, 0.17332387096774193, 0.20253064516129032, 0.9327],
    "27": [0.046935483870967735, 0.11264516129032257, 0.13141935483870967, 0.291],
    "5": [0.0338258064516129, 0.0649832258064516, 0.18961290322580643, 0.24154193548387093],
    "7": [0.006490322580645161, 0.10384516129032258, 0.1168258064516129, 0.162258064516129]
  },
  "n_classes_per_cons": [2, 4, 0],
  "n_labels": 4,
  "n_labels_per_cons": [2, 4, 3],
  "rules": [
    [
      [
        ["5", 1],
        ["14", 0]
      ],
      [0.0, 1.0, 1.0]
    ],
    [
      [
        ["24", 0]
      ],
      [1.0, 1.0, 2.0]
    ],
    [
      [
        ["27", 1],
        ["25", 3]
      ],
      [1.0, 0.0, 2.0]
    ],
    [
      [
        ["24", 1],
        ["0", 2],
        ["7", 0]
      ],
      [0.0, 0.0, 0.0]
    ],
    [
      [
        ["1", 3]
      ],
      [0.0, 3.0, 0.0]
    ]
  ],
  "vars_range": {
    "0": [6.981, 28.11],
    "1": [9.71, 39.28],
    "14": [0.001713, 0.03113],
    "24": [0.07117, 0.2226],
    "25": [0.02729, 0.9327],
    "27": [0.0, 0.291],
    "5": [0.02344, 0.3454],
    "7": [0.0, 0.2012]
  },
  "version": 1
} 
    """
    converted_fis = TffConverter.to_fis(tff_str)
