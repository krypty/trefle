import pyfuge_c


class NativeCocoEvaluator:
    def __init__(
        self,
        n_vars,
        n_rules,
        n_max_vars_per_rule,
        n_bits_per_mf,
        n_true_labels,
        n_lv_per_ind,
        n_bits_per_ant,
        n_cons,
        n_bits_per_cons,
        n_bits_per_label,
    ):
        self._fiseval = pyfuge_c.FISCocoEvalWrapper(
            n_vars,
            n_rules,
            n_max_vars_per_rule,
            n_bits_per_mf,
            n_true_labels,
            n_lv_per_ind,
            n_bits_per_ant,
            n_cons,
            n_bits_per_cons,
            n_bits_per_label,
        )

    def predict_native(self, ind_sp1: str, ind_sp2: str):
        y_pred = self._fiseval.bind_predict(ind_sp1, ind_sp2)

        return y_pred


if __name__ == "__main__":
    import numpy as np
    from pyfuge.evo.experiment.coco.coco_individual import CocoIndividual, MFShape
    import random

    np.random.seed(2)
    random.seed(2)

    X_train = np.array(
        [
            # [-2, 70, 3, 4],
            # [144, 52, 5, -3],
            # [2.4, 5, 5, 1.33],
            [-2, 70, 3, 4, 144, 52, 5, -3, 1.33],
            [10, 100, 2, 4, 6, 67, 2, -212, 9.33],
            [-7, 60, -1.93, 1.11, 3.45, -1.3, 1.4, 0, 1.111],
        ]
    )

    # y_train = np.array([0, 1])
    y_train = np.array([[0, 2, 3.2], [1, 1, 4.4], [0, 0, 1.3]])
    coco_ind = CocoIndividual(
        X_train=X_train,
        y_train=y_train,
        # problem_type=ProblemType.CLASSIFICATION,
        n_rules=3,
        n_classes_per_cons=[2, 3, 0],
        n_max_vars_per_rule=4,
        n_labels_per_mf=2,
        p_positions_per_lv=32,  # 5 bits
        dc_padding=1,
        mfs_shape=MFShape.TRI_MF,
        n_lv_per_ind_sp1=None,
    )

    ind_sp1 = coco_ind.generate_sp1()
    ind_sp2 = coco_ind.generate_sp2()

    res = coco_ind.predict((ind_sp1, ind_sp2))
    print(res)
