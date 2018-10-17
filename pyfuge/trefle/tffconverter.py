import json
import os
from pyfuge_c import TrefleFIS

from pyfuge.fs.core.fis.singleton_fis import SingletonFIS
from pyfuge.fs.core.lv.p_points_lv import PPointsLV
from pyfuge.fs.core.rules.fuzzy_rule import FuzzyRule
from pyfuge.fs.view.lv_viewer import LinguisticVariableViewer


class TffJsonToSingletonFIS:
    SUPPORTED_VERSION = 1

    def __init__(self, tff_str):
        self._tff_str = tff_str
        self._jfis = json.loads(self._tff_str)

    def convert(self):
        self._ensure_version()

        lvs = self._parse_lvs()

        for name, lv in lvs.items():
            print(name)
            LinguisticVariableViewer(lv).show()

        # fr = FuzzyRule()
        # rules = []
        # default_rule = []
        #
        # fis = SingletonFIS(
        #     rules=rules,
        #     default_rule=default_rule
        # )
        # print(jfis)

        # TODO return SingletonFIS

    def _ensure_version(self):
        if self._jfis["version"] != self.SUPPORTED_VERSION:
            raise ValueError(
                "Unsupported tff version! Currently supported: {}".format(
                    self.SUPPORTED_VERSION
                )
            )

    def _parse_lvs(self):
        def scale_back_var(p_pos, _var_range):
            return [(_var_range[1] - _var_range[0]) * p + _var_range[0] for p in p_pos]

        lvs = self._jfis["linguistic_variables"]
        vars_range = self._jfis["vars_range"]

        for v_idx, p_positions in lvs.items():
            lvs[v_idx] = scale_back_var(p_positions, vars_range[v_idx])

        return {name: PPointsLV(name, p_pos) for name, p_pos in lvs.items()}


class TffConverter:
    @staticmethod
    def to_fis(tff_str: str):
        """

        :param tff_str: if tff_str is a valid file path, this latter
        will be read and parsed. Otherwise tff_str will be interpreted as
        a json str and parsed directly.
        :return: a SingletonFIS instance
        """
        TffJsonToSingletonFIS(tff_str).convert()
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
        from pyfuge.trefle.tffconverter import TffConverter as tamere

        # obj = TrefleFIS(30)
        # obj.predict()
        #
        # obj1 = TrefleFIS.from_tff("sjdlkaj")
        # obj1.predict()
        #
        # obj2 = TrefleFIS.from_tff_file("sjdlkaj")
        # obj2.predict()

        tff_str = r"""
       {"default_rule":[1.0,2.0,1.0],"linguistic_variables":{"0":[13.796806451612902,14.478387096774192,16.523129032258062,25.38367741935484],"1":[25.925806451612903,29.741290322580646,32.60290322580645,34.51064516129033],"14":[0.004559806451612903,0.014998096774193549,0.02164064516129032,0.030181064516129036],"24":[0.08582451612903226,0.09070935483870968,0.12978806451612904,0.13955774193548387],"25":[0.11491032258064515,0.17332387096774193,0.20253064516129032,0.9327],"27":[0.046935483870967735,0.11264516129032257,0.13141935483870967,0.291],"5":[0.0338258064516129,0.0649832258064516,0.18961290322580643,0.24154193548387093],"7":[0.006490322580645161,0.10384516129032258,0.1168258064516129,0.162258064516129]},"n_classes_per_cons":[2,4,0],"n_labels":4,"n_labels_per_cons":[2,4,3],"rules":[[[["5",1],["14",0]],[0.0,1.0,1.0]],[[["24",0]],[1.0,1.0,2.0]],[[["27",1],["25",3]],[1.0,0.0,2.0]],[[["24",1],["0",2],["7",0]],[0.0,0.0,0.0]],[[["1",3]],[0.0,3.0,0.0]]],"vars_range":{"0":[6.981,28.11],"1":[9.71,39.28],"14":[0.001713,0.03113],"24":[0.07117,0.2226],"25":[0.02729,0.9327],"27":[0.0,0.291],"5":[0.02344,0.3454],"7":[0.0,0.2012]},"version":1} 
        """
        fis = tamere.to_fis(tff_str)
