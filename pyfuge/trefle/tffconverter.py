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

        lvs = self._parse_lvs()

        cons_lvs = self._parse_cons_lvs()

        rules = self._parse_rules(lvs, labels, cons_lvs, cons_labels)

        default_rule = self._parse_default_rule(cons_lvs, cons_labels)

        fis = SingletonFIS(rules, default_rule)

        return fis

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
    import numpy as np

    # run use_trefle.py before!
    tff_str = open("/tmp/temp.tff").read()
    trefle_fis = TffConverter.to_trefle_fis(tff_str)
    X_test = np.load("/tmp/X_test.npy")
    y_pred_expected = np.load("/tmp/y_pred.npy")

    y_pred = trefle_fis.predict(X_test)

    # print(y_pred)
    # print("-----------")
    # print(y_pred_expected)
    #
    if np.allclose(y_pred, y_pred_expected):
        print("YEAHHH")
    else:
        print("nope...")
