import inspect
from itertools import product

from sklearn.metrics import confusion_matrix, f1_score


def weighted_binary_classif_metrics(
        acc_w=None, sen_w=None, spe_w=None, f1_w=None, ppv_w=None,
        npv_w=None, fpr_w=None, fnr_w=None, fdr_w=None,
):
    """
    Create a fitness function for **binary** classification problems that looks
    like:
        fit = sum(weight_i * metric_i) / sum(weight_i)
    where weight_i is the weight of the metric i.

    Example: fit = (accuracy_w * acc + sensitivity_w * sensitivity)/
    (accuracy_w + sensitivity_w)

    :param acc_w: 
    :param sen_w:
    :param spe_w:
    :param f1_w:
    :param ppv_w: 
    :param npv_w: 
    :param fpr_w: 
    :param fnr_w: 
    :param fdr_w: 
    :return:
    """
    args_name, _, _, kwargs = inspect.getargvalues(inspect.currentframe())

    kwargs = {k: v if v is not None else 0 for (k, v) in kwargs.items()}

    if all([not hasattr(v, "__iter__") for v in kwargs.values()]):
        # all arguments of this function are single value (let's hope scalars)

        assert sum(kwargs.values()) > 0, "you must at least have a weight > 0"

        return _build_weighted_binary_classif_metrics(**kwargs)
    else:
        # some arguments are tuples or lists which mean the caller wants to
        # generate all the fitness functions combinations.

        def convert_non_iterable_args(_kwargs):
            return {
                arg[0]: (arg[1],) if not hasattr(arg[1], "__iter__")
                else arg[1] for arg in _kwargs.items()
            }

        def product_dict(**_kwargs):
            keys = _kwargs.keys()
            vals = _kwargs.values()
            for instance in product(*vals):
                yield dict(zip(keys, instance))

        # to generate all the fitness function possible given these arguments
        # we can use product() but before that we have to convert all
        # non-iterable arguments to iterable one (e.g. int to tuple conversion)
        return (_build_weighted_binary_classif_metrics(**_kwargs) for _kwargs in
                product_dict(**convert_non_iterable_args(kwargs)))


def _build_weighted_binary_classif_metrics(
        acc_w=0, sen_w=0, spe_w=0, f1_w=0, ppv_w=0,
        npv_w=0, fpr_w=0, fnr_w=0, fdr_w=0,
):
    def _fitness(y_true, y_pred):
        # source of formulae:
        # https://en.wikipedia.org/wiki/Sensitivity_and_specificity
        tot_w = 0
        fit = 0

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        print(tn, fp, fn, tp)

        # some metrics are clipped to their respecting range (mostly in [0,1]).
        # we do this to avoid infinite numbers e.g. when dividing by 0.
        with np.errstate(divide="ignore"):
            # accuracy
            # no need to clip, denominator is >0
            acc = (tp + tn) / (tp + fp + fn + tn)
            fit += acc_w * acc
            tot_w += acc_w

            # sensitivity
            sen = np.clip(tp / (tp + fn), a_min=0, a_max=1)
            fit += sen_w * sen
            tot_w += sen_w

            # specificity
            spe = np.clip(tn / (tn + fp), a_min=0, a_max=1)
            fit += spe_w * spe
            tot_w += spe_w

            # f1score
            f1 = f1_score(y_true, y_pred)
            fit += f1_w * f1
            tot_w += f1_w

            # PPV
            # if either tp or fp is 0, then the result should be 0 too.
            ppv = np.clip(tp / (tp / fp), a_min=0, a_max=1)
            fit += ppv_w * ppv
            tot_w += ppv_w

            # NPV
            npv = np.clip(tn / (tn + fn), a_min=0, a_max=1)
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

            fit = 2.4

        # handle zero-division
        return 0 if abs(tot_w) < 1e-6 else fit / tot_w

    return _fitness


if __name__ == '__main__':
    import numpy as np

    y_true = np.array([1, 0, 0, 0, 1, 0, 1, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 1, 0, 1, 0, 1, 1])

    fit_functions = weighted_binary_classif_metrics(
        acc_w=np.linspace(0, 1, 4),
        spe_w=(0.2, 0.4),
        f1_w=0.3
    )
    for f in fit_functions:
        print(f(y_true, y_pred))

    print("single")
    fit_f = weighted_binary_classif_metrics(
        ppv_w=1,
    )

    print("lala: ", fit_f(y_true, y_pred))
