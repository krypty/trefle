import inspect
import sys
from itertools import product
from warnings import catch_warnings, filterwarnings

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import confusion_matrix, f1_score, mean_squared_error

from trefle.fitness_functions.output_thresholder import round_to_cls


def weighted_binary_classif_metrics(
    acc_w=None,
    sen_w=None,
    spe_w=None,
    f1_w=None,
    ppv_w=None,
    npv_w=None,
    fpr_w=None,
    fnr_w=None,
    fdr_w=None,
    mse_w=None,
):
    """
    Create a fitness function for **binary** classification problems that looks
    like:
        fit = sum(weight_i * metric_i) / sum(weight_i)
    where weight_i is the weight of the metric i.

    Example: fit = (accuracy_w * acc + sensitivity_w * sensitivity)/
    (accuracy_w + sensitivity_w)

    All weights are expected to be positive numbers.

    :param acc_w:
    :param sen_w:
    :param spe_w:
    :param f1_w:
    :param ppv_w:
    :param npv_w:
    :param fpr_w:
    :param fnr_w:
    :param fdr_w:
    :param mse_w:
    :return:
    """
    args_name, _, _, kwargs = inspect.getargvalues(inspect.currentframe())

    kwargs = {k: v if v is not None else 0 for (k, v) in kwargs.items()}

    if all([not hasattr(v, "__iter__") for v in kwargs.values()]):
        # all arguments of this function are single value (let's hope scalars)

        if sum(kwargs.values()) <= 0:
            raise ValueError("you must at least have a weight > 0")

        return _build_weighted_binary_classif_metrics(**kwargs)
    else:
        # some arguments are tuples or lists which mean the caller wants to
        # generate all the fitness functions combinations.

        def convert_non_iterable_args(_kwargs):
            return {
                arg[0]: (arg[1],) if not hasattr(arg[1], "__iter__") else arg[1]
                for arg in _kwargs.items()
            }

        def product_dict(**_kwargs):
            keys = _kwargs.keys()
            vals = _kwargs.values()
            for instance in product(*vals):
                yield dict(zip(keys, instance))

        # to generate all the fitness function possible given these arguments
        # we can use product() but before that we have to convert all
        # non-iterable arguments to iterable one (e.g. int to tuple conversion)
        return list(
            _build_weighted_binary_classif_metrics(**_kwargs)
            for _kwargs in product_dict(**convert_non_iterable_args(kwargs))
        )


def _build_weighted_binary_classif_metrics(
    acc_w=0,
    sen_w=0,
    spe_w=0,
    f1_w=0,
    ppv_w=0,
    npv_w=0,
    fpr_w=0,
    fnr_w=0,
    fdr_w=0,
    mse_w=0,
):
    _, _, _, _kwargs = inspect.getargvalues(inspect.currentframe())

    class Fitness:
        def __init__(self):
            if sum(_kwargs.values()) <= 0:
                print(
                    "warning: all metrics have a weight of 0 which"
                    " will lead to a 0 fitness value",
                    file=sys.stderr,
                )

        @staticmethod
        def weights():
            return _kwargs

        def __repr__(self):
            return str(self.weights())

        def __getitem__(self, item):
            return _kwargs[item]

        def __call__(self, *args, **kwargs):
            # y_true = kwargs["y_true"]
            # y_pred = kwargs["y_pred"]

            # mimic fitness_function_signature
            y_true = args[0]
            y_pred = args[1]

            return self._fitness(y_true, y_pred)

        @staticmethod
        def _fitness(y_true, y_pred):
            # source of formulae:
            # https://en.wikipedia.org/wiki/Sensitivity_and_specificity
            tot_w = 0
            fit = 0

            y_pred_bin = round_to_cls(y_pred, n_classes=2)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()

            # some metrics are set to 0 (np.nan_to_num) because we want to
            # avoid infinite numbers e.g. when dividing by 0. Oddly, we need
            # to filter "invalid" too to handle division errors. Be careful,
            # this only works if the metrics is a "the higher the
            # better"-metric
            with np.errstate(divide="ignore", invalid="ignore"):
                # accuracy
                # no need to clip, denominator is >0
                acc = (tp + tn) / (tp + fp + fn + tn)
                fit += acc_w * acc
                tot_w += acc_w

                # sensitivity
                sen = np.nan_to_num(tp / (tp + fn))
                fit += sen_w * sen
                tot_w += sen_w

                # specificity
                spe = np.nan_to_num(tn / (tn + fp))
                fit += spe_w * spe
                tot_w += spe_w

                # f1score, ignore ill-defined value, it will be set to 0
                with catch_warnings():
                    filterwarnings("ignore", category=UndefinedMetricWarning)
                    f1 = f1_score(y_true, y_pred_bin)
                    fit += f1_w * f1
                    tot_w += f1_w

                # PPV
                # if either tp or fp is 0, then the result should be 0 too.
                ppv = np.nan_to_num(tp / (tp + fp))
                fit += ppv_w * ppv
                tot_w += ppv_w

                # NPV
                # if either tp or fp is 0, then the result should be 0 too.
                # note: we don't reuse PPV value because it could have been set
                # to 0 due to nan result.
                npv = np.nan_to_num(tn / (tn + fn))
                fit += npv_w * npv
                tot_w += npv_w

                # FPR
                fpr = 1 - spe
                fit += fpr_w * fpr
                tot_w += fpr_w

                # FNR
                fnr = 1 - sen
                fit += fnr_w * fnr
                tot_w += fnr_w

                # FDR
                fdr = 1 - ppv
                fit += fdr_w * fdr
                tot_w += fdr_w

                # MSE
                mse = -mean_squared_error(y_true, y_pred)
                fit += mse_w * mse
                tot_w += mse_w

            # handle zero-division
            return 0 if abs(tot_w) < 1e-6 else fit / tot_w

    return Fitness()


if __name__ == "__main__":
    # WIP do not use yet

    y_true = np.array([1, 0, 0, 0, 1, 0, 1, 0, 1, 1])
    y_pred = np.array([1.9, 0.22, 0.34, 0.32, 0.9, 0.1, 0.88, 0.1, 1.0, 0.81])
    y_pred_bin = np.array([0, 0, 0, 0, 1, 0, 1, 0, 1, 1])

    fit_functions = weighted_binary_classif_metrics(
        acc_w=1.0,
        # acc_w=np.linspace(0, 1, 2),
        spe_w=(0.2, 0.4),
    )

    for f in fit_functions:
        print("function weights", f)
        print(f(y_true, y_pred_bin))

    print("single")
    fit_f = weighted_binary_classif_metrics(ppv_w=1)

    print("fit_f: ", fit_f(y_true, y_pred))
