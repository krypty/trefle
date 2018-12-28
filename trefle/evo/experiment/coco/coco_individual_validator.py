from math import log, ceil

import numpy as np

from trefle.evo.helpers.fuzzy_labels import LabelEnum, LElem


def ensure(condition, msg="Assertion is incorrect"):
    if not condition:
        raise ValueError(msg)


class CocoIndividualValidator:
    """
    This **highly** coupled class with CocoIndividual ensure that a couple of
    requirements is matched.
    """

    def __init__(self, coco_ind):
        self._coco_ind = coco_ind

    def validate(self):
        ensure(self._coco_ind._n_max_vars_per_rule > 0, "max_vars_per_rule > 0")
        ensure(self._coco_ind._n_max_vars_per_rule <= self._coco_ind._n_vars)

        ensure(self._coco_ind._dc_weight >= 0, "negative padding does not make sense")
        ensure(
            log(self._coco_ind._p_positions_per_lv, 2)
            == ceil(log(self._coco_ind._p_positions_per_lv, 2)),
            "p_positions_per_lv must be a multiple of 2",
        )

        ensure(
            self._coco_ind._p_positions_per_lv >= self._coco_ind._n_true_labels,
            (
                "You must have at least as many p_positions as the n_labels_per_mf "
                "you want to use "
            ),
        )

        ensure(
            issubclass(self._coco_ind._n_labels_cons, LabelEnum),
            "You must use a subclass of LabelEnum (e.g. Label4) for n_labels_per_cons",
        )

        # Validate the number of classes per consequent
        n_classes_per_cons_in_y = np.apply_along_axis(
            lambda c: len(np.unique(c)), arr=self._coco_ind._y, axis=0
        ).reshape(-1)
        # force to have an array with a shape

        msg = (
            "the number of consequents indicated in n_class_per_cons does not "
            "match what was computed on y_train. from user: {}, computed: {}"
        )

        ensure(len(self._coco_ind._n_classes_per_cons) == len(n_classes_per_cons_in_y))

        n_cls_per_cons_zeroed = n_classes_per_cons_in_y.copy()
        # we don't want to compare the number of classes for continuous vars
        n_cls_per_cons_zeroed[self._coco_ind._n_classes_per_cons == 0] = 0
        ensure(
            np.array_equal(
                self._coco_ind._n_classes_per_cons.flatten(),
                n_cls_per_cons_zeroed.flatten(),
            ),
            msg.format(self._coco_ind._n_classes_per_cons, n_classes_per_cons_in_y),
        )

        ensure(
            all([c >= 0 for c in self._coco_ind._n_classes_per_cons]),
            "n_classes values must be positive in n_classes_per_cons",
        )

        mask = n_classes_per_cons_in_y == self._coco_ind._n_classes_per_cons
        ensure(
            all(mask[self._coco_ind._n_classes_per_cons != 0]),
            "the n_classes per consequent does not match with what found on X_train",
        )

        ensure(
            (2 ** self._coco_ind._n_bits_per_lv >= self._coco_ind._n_max_vars_per_rule),
            "n_lv_per_ind_sp1 must be at least equals to n_max_vars_per_rule",
        )

        ensure(
            issubclass(self._coco_ind._n_labels_cons, LabelEnum),
            "n_labels _cons must an instance of a subclass of LabelEnum",
        )

        ensure(
            self._coco_ind._default_cons.shape[0] == self._coco_ind._n_cons,
            (
                "default_cons's shape doesn't match the number of "
                "consequents retrieved using y_train"
            ),
        )

        # we check if a cons is either an int or the same class as
        # self._coco_ind._n_labels_cons. For the latter we check like that instead of
        # check issubclass because we do care that the label values of both
        # n_labels_cons and default_cons are the same (e.g. if
        # n_labels_cons's LOW = 0, then default_cons' LOW = 0 too)
        are_labels_or_int = [
            isinstance(c, (int, np.integer, LElem))
            for c in self._coco_ind._default_cons
        ]

        ensure(
            all(are_labels_or_int),
            (
                "The default rule must only contain classes or labels"
                " i.e. integer numbers. If a label is provide like LabelX.LOW"
                " make sure that the X in LabelX is the same for both"
                " n_labels_cons (currently set to {})"
                " and default_cons".format(self._coco_ind._n_labels_cons.__name__)
            ),
        )

        def can_default_cons_fit_in_cons():

            for default_cons, n_labels_for_cons in zip(
                self._coco_ind._default_cons, self._coco_ind._cons_n_labels
            ):
                if isinstance(default_cons, LElem):
                    # regression: make sure both a and b are LabelX
                    # (not LabelX and LabelY)
                    yield len(default_cons) == n_labels_for_cons
                else:
                    # classification: make sure the specified class can fit in
                    # the consequent
                    yield default_cons < n_labels_for_cons

        ensure(
            all(can_default_cons_fit_in_cons()),
            (
                "Make sure that the default rule contains valid classes/labels \n"
                "i.e. label is in [0, n_classes-1] or in case of regression in \n"
                "[0, n_labels-1] Check the parameters 'n_labels_per_cons' and \n"
                " 'default_cons' .\n"
                "Expected: ({}) < {}".format(
                    self._coco_ind._default_cons, self._coco_ind._cons_n_labels
                )
            ),
        )
