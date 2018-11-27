from trefle_engine import FISCocoEvalWrapper

import numpy as np


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
        self._eval_wrapper = FISCocoEvalWrapper(
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
        if other_X is None:
            y_pred = self._eval_wrapper.bind_predict(ind_sp1, ind_sp2)
        else:
            y_pred = self._eval_wrapper.bind_predict(ind_sp1, ind_sp2, other_X)
        return y_pred

    def print_ind(self, ind_sp1: str, ind_sp2: str):
        self._eval_wrapper.print_ind(ind_sp1, ind_sp2)

    def to_tff(self, ind_sp1: str, ind_sp2: str):
        return self._eval_wrapper.to_tff(ind_sp1, ind_sp2)
