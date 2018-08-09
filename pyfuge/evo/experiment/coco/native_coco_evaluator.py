import pyfuge_c


class NativeCocoEvaluator:

    # def __init__(self, ind_n, observations, n_rules, max_vars_per_rule,
    #              n_labels, n_consequents, default_rule_cons, vars_ranges):
    def __init__(self, a, b):
        self._fiseval = pyfuge_c.FISCocoEvalWrapper(
            a, b
            # ind_n, observations, n_rules, max_vars_per_rule, n_labels,
            # n_consequents, default_rule_cons, vars_ranges
        )

    def predict_native(self, ind_sp1: str, ind_sp2: str):
        y_pred = self._fiseval.bind_predict(ind_sp1, ind_sp2)

        return y_pred


if __name__ == '__main__':
    nce = NativeCocoEvaluator(a=22, b=2345)
    res = nce.predict_native(ind_sp1="lala", ind_sp2="toto")
    print(res)
