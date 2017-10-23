from core.fis.fis import FIS
from core.membership_functions.free_shape_mf import FreeShapeMF


class SingletonFIS(FIS):
    def _aggregate(self, rules_implicated_cons):
        aggregated_consequents = {}
        for index, (out_v_name, out_v_mf) in enumerate(
                rules_implicated_cons.items()):
            # print("yolo", out_v_name, out_v_mf)
            # print(out_v_mf[0].mf_values[0])

            numerator = 0
            denominator = 0
            for i, rule in enumerate(chain(self._rules, [self._default_rule])):
                cons_implicated_value = out_v_mf[i].mf_values[0]
                label = rule.consequents[index].lv_value
                print(label)
                rule_act_value = \
                    rule.consequents[index].lv_name.ling_values[
                        label].in_values[0]

                print(cons_implicated_value, rule_act_value)

                numerator += cons_implicated_value * rule_act_value
                denominator += cons_implicated_value

            aggregated_consequents[out_v_name] = FreeShapeMF(
                in_values=[numerator / float(denominator)], mf_values=[1])

        print(aggregated_consequents)
        return aggregated_consequents

    def _defuzzify(self):
        self._defuzzified_outputs = {k: v.in_values[0] for k, v in
                                     self._aggregated_consequents.items()}
        return self._defuzzified_outputs
