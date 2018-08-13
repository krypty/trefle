import pyfuge_c


class NativeCocoEvaluator:

    # def __init__(self, ind_n, observations, n_rules, max_vars_per_rule,
    #              n_labels, n_consequents, default_rule_cons, vars_ranges):
    def __init__(self,
                 n_bits_per_mf,
                 n_true_labels,
                 n_mf_per_ind
                 ):
        self._fiseval = pyfuge_c.FISCocoEvalWrapper(
            n_bits_per_mf,
            n_true_labels,
            n_mf_per_ind
            # ind_n, observations, n_rules, max_vars_per_rule, n_labels,
            # n_consequents, default_rule_cons, vars_ranges
        )

    def predict_native(self, ind_sp1: str, ind_sp2: str):
        y_pred = self._fiseval.bind_predict(ind_sp1, ind_sp2)

        return y_pred


if __name__ == '__main__':
    import numpy as np
    from pyfuge.evo.experiment.coco.coco_individual import CocoIndividual, \
        ProblemType, MFShape
    import random

    np.random.seed(2)
    random.seed(2)

    X_train = np.array([
        [-2, 70, 1.33],
        [10, 100, 9.33],
        [-7, 60, -1.93],
    ])

    y_train = np.array([0, 0, 1])
    coco_ind = CocoIndividual(
        X_train=X_train,
        y_train=y_train,
        problem_type=ProblemType.CLASSIFICATION,
        n_rules=2,
        n_max_vars_per_rule=4,
        n_labels_per_mf=2,
        p_positions_per_lv=32,  # 5 bits
        dc_padding=1,
        mfs_shape=MFShape.TRI_MF,
        p_positions_per_cons=16,  # 4 bits
        n_lv_per_ind_sp1=None
    )

    ind_sp1 = coco_ind.generate_sp1()
    ind_sp2 = coco_ind.generate_sp2()

    res = coco_ind.predict((ind_sp1, ind_sp2))
    print(res)
