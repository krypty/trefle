import pyfuge_c

import numpy as np

from pyfuge.evo.helpers.fuzzy_labels import Label9


class NativeCocoEvaluator:
    def __init__(
        self,
        X_train: np.array,
        n_vars: int,
        n_rules: int,
        n_max_vars_per_rule: int,
        n_bits_per_mf: int,
        n_true_labels: int,
        n_bits_per_lv: int,
        n_bits_per_ant: int,
        n_cons: int,
        n_bits_per_cons: int,
        n_bits_per_label: int,
        dc_weight: int,
        cons_n_labels: np.array,
        n_classes_per_cons: np.array,
        default_cons: np.array,
        vars_range: np.array,
        cons_range: np.array,
    ):
        self._fiseval = pyfuge_c.FISCocoEvalWrapper(
            X_train,
            n_vars,
            n_rules,
            n_max_vars_per_rule,
            n_bits_per_mf,
            n_true_labels,
            n_bits_per_lv,
            n_bits_per_ant,
            n_cons,
            n_bits_per_cons,
            n_bits_per_label,
            dc_weight,
            cons_n_labels,
            n_classes_per_cons,
            default_cons,
            vars_range,
            cons_range,
        )

    def predict_native(self, ind_sp1: str, ind_sp2: str, other_X: np.array = None):
        # yolo = np.array(
        #     [[1, 2, 3, 4, 5], [10, 11, 12, 13, 14], [20, 21, 22, 23, 24]]
        # ).astype(np.float)
        # y_pred = self._fiseval.bind_predict(ind_sp1, ind_sp2, yolo)
        #
        if other_X is None:
            y_pred = self._fiseval.bind_predict(ind_sp1, ind_sp2)
        else:
            y_pred = self._fiseval.bind_predict(ind_sp1, ind_sp2, other_X)
        return y_pred

    # def predict_native(self, ind_sp1: str, ind_sp2: str):
    #     y_pred = self._fiseval.bind_predict(ind_sp1, ind_sp2)
    #     return y_pred

    def print_ind(self, ind_sp1: str, ind_sp2: str):
        self._fiseval.print_ind(ind_sp1, ind_sp2)

    def to_tff(self, ind_sp1: str, ind_sp2: str):
        return self._fiseval.to_tff(ind_sp1, ind_sp2)


if __name__ == "__main__":
    import numpy as np
    from pyfuge.evo.experiment.coco.coco_individual import CocoIndividual, MFShape
    import random

    np.random.seed(7)
    random.seed(7)

    # X_train, y_train = load_wine(return_X_y=True)
    # y_train = y_train.reshape(-1, 1)
    # print(y_train)

    X_train = np.array(
        [
            [-2, 70, 3, 4],
            [14, 52, 5, -3],
            [2.4, 5, 5, 1.33],
            [7.4, 15, 2, 9.33],
            # [-2, 70, 3, 4, 144, 52, 5, -3, 1.33],
            # [10, 100, 2, 4, 6, 67, 2, -212, 9.33],
            # [-7, 60, -1.93, 1.11, 3.45, -1.3, 1.4, 0, 1.111],
        ]
    )

    # y_train = np.array([0, 1, 1, 1, 0, 0, 0])
    y_train = np.array([[0, 2, 3.2], [1, 1, -4.4], [0, 0, 1.3], [0, 1, 3.54]])
    coco_ind = CocoIndividual(
        X_train=X_train,
        y_train=y_train,
        # problem_type=ProblemType.CLASSIFICATION,
        n_rules=3,
        # n_classes_per_cons=[2],
        n_classes_per_cons=[2, 3, 0],
        n_max_vars_per_rule=4,
        n_labels_per_mf=2,
        p_positions_per_lv=32,  # 5 bits
        dc_weight=1,
        mfs_shape=MFShape.TRI_MF,
        n_lv_per_ind_sp1=5,
        n_labels_per_cons=Label9,
        # default_cons=[1],
        # TODO: handle me ! Do not forget to minmax normed and scale back. Here 3.2 should be an integer representing a cons label
        # FIXME: this is wrong. Should be a label not a real value! But y_pred should be scaled back!
        # default_cons=[0, 1, 3.2],
        default_cons=[0, 1, Label9.VERY_VERY_VERY_HIGH],
    )

    # from time import time
    #
    # t0 = time()
    ind_sp1 = coco_ind.get_ind_sp1_class()()
    print("ind_sp1", ind_sp1.bits)
    ind_sp2 = coco_ind.get_ind_sp2_class()()
    print("ind_sp2", ind_sp2.bits, " ", len(ind_sp2.bits))
    # t0 = time() - t0
    # print("time", t0 * 1000, " ms")

    # t0 = time()
    res = coco_ind.predict((ind_sp1, ind_sp2))
    # t0 = time() - t0
    # print("time", t0 * 1000, " ms")
    print(res)

    coco_ind.print_ind((ind_sp1, ind_sp2))

    # print("predict new X")
    # res = coco_ind.predict(
    #     (ind_sp1, ind_sp2), np.asarray([[0, 1, 3.444, 4, 5, 6, 7, 8, 88]])
    # )
    # print(res)
