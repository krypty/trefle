from enum import Enum
from math import ceil, log
from random import randint
from typing import List

import numpy as np
from bitarray import bitarray

from pyfuge.evo.experiment.coco.native_coco_evaluator import NativeCocoEvaluator
from pyfuge.evo.helpers.fis_individual import FISIndividual, Clonable
from pyfuge.evo.helpers.fuzzy_labels import LabelEnum, Label3

"""
Convention: a variable to represent the number of bits needed for a matrix
must start with n_bits_XXX.
It is defined like: n_bits_XXX = n_rows * n_cols * n_bits_per_element
"""


class ProblemType(Enum):
    CLASSIFICATION = 0
    REGRESSION = 1


class MFShape(Enum):
    TRI_MF = 0
    TRAP_MF = 1


def minmax_norm(X_train):
    print("warning: please implement this!")
    norm_args = "yolo"
    return X_train, norm_args


class CocoIndividual(FISIndividual, Clonable):
    """
    This class creates two individuals i.e. one to represent the membership
    functions and the other to represent the rules, called respectively
    specie 1 (sp1) and specie 2 (sp2)

    Reference: Carlos Pena's thesis Table 3.1 chapter Application Example The
    Iris Problem

    **Specie 1 - the MFs**
    The shape of sp1 is the following:
        [ v0p0, v0p1, ..., v0pK, v1p0, v1p2, ..., v1pK, ...]


    Meaning: vipj is the j-th point (or p position) of the i-th variable.
    For example, for triangular MF (triMF) the value v4p1 is represented by:


        Membership functions for variable 4
      ^
      | low      medium           high
    1 |XXXXX       X          XXXXXXXXXXXX
      |     X     X  X       XX
      |      X   X    X    XX
      |       X X      XX X
      |       XX        XXX
      |      X  X     XX   XX
      |     X    X XX       XX
      |    X       X          XX
    0 +-------------------------------------->
           p0     p1          p2

    In vipK the "K" defines the number of labels a linguistic variable has. For
    example, if K=3 then the linguistic variables created will use 3 membership
    functions to represent the LOW, MEDIUM and HIGH labels. This "k" letter is
    also called n_true_labels in the code.

    The vipj value is in the interval [min(var_i), max(var_i)] computed from the
    dataset (i.e. X_train). A vipj value is represented as a binary number to
    let the evolutionary algorithm manipulate it.

    The resolution (i.e. the number of possible values that a p position can
    take is defined by the p_positions_per_lv. So the number of bits needed to
    represented the desired p_positions_per_lv is:
    n_bits_per_mf = ceil(log2(p_positions_per_lv))

    In total there are:
        n_bits_per_mf = ceil(log2(p_positions_per_lv))
        n_bits_for_all_vars = n_vars * n_bits_per_mf * n_true_labels
        TODO: n_vars or n_max_vars_per_rule ?




    **Specie 2 - the rules** There are two parameters Aij and Cij.
    The shape of sp2 is the following:
        [ r0a0, r0a1,...r0aN, r1a0, r1a1,..rMaN, r0mf0, r0mf1, r0mfN..r1mf0,
        r1mf1,..rMmfN, r0c0, r0c1,...r0cJ, r1c0, r1c1,...rMcJ ]

    where:
        M = n_rules
        N = n_max_vars_per_rule
        J = n_consequents i.e. number of output variables that are NOT mutually
            exclusive

    Aij parameter indicates which antecedents/variables are used by the
    model/fis for all its rules.

    It consists of two "matrices" let's call them r_ants and r_mfs.

    r_ants is a NxM matrix like this:

        ^             ||<----- n_max_vars_per_rule ----->|
        |     |-------||------|------|------|-----|------|
    n_rules   | rule0 || r0a0 | r0a1 | r0a2 | ... | r0aN |
        |     | rule1 || r1a0 | r1a1 | r1a2 | ... | r1aN |
        v     | ruleM || rMa0 | rMa1 | rMa2 | ... | rMaN |

    A single element e.g. r1a2 is a link/"pointer" to the variable index/number
    to use for the 3rd antecedent (a2) of the 2nd rule (r1). A single element
    is in the interval [0, n_vars -1] where n_vars is deduced from X_train.


    r_mfs is a NxM matrix like this:

        ^             ||<------- n_max_vars_per_rule ------->|
        |     |-------||-------|-------|-------|-----|-------|
    n_rules   | rule0 || r0mf0 | r0mf1 | r0mf2 | ... | r0mfN |
        |     | rule1 || r1mf0 | r1mf1 | r1mf2 | ... | r1mfN |
        v     | ruleM || rMmf0 | rMmf1 | rMmf2 | ... | rMmfN |

    A single element e.g. r1mf2 is a link/"pointer" to the membership function
    index (by index we mean for example 0 for LOW, 1 for MEDIUM, 2 for HIGH, ...
    last for DONT_CARE) to use for the variable/antecedent r1a2 of the r_ants
    "matrix". For example if r1a2=6, r1mf2=2 and n_true_labels=4 (i.e. VERY LOW,
    LOW, MEDIUM, HIGH) then it would mean that the 3rd antecedent (a2) of the
    2nd rule (r1) is "5th variable is MEDIUM". (5th because of "=6" and MEDIUM
    because of mf=2 with n_true_labels=4). A single element is in the interval
    [0, (n_true_labels + dc_weight)-1]. With dc_weight=1 all mf have the
    TODO change dc_weight comment
    same probability to be chosen. To increase the probability to chose/select
    a DC, increase dc_weight.


    The other parameter, Cij indicates which consequents class
    (in a classification problem) or a real value (in a regression problem) are
    used by the model/fis for all its rules.

    It consists of a single "matrix" let's call it r_cons.

    r_cons is a NxJ matrix like:


        ^             ||<-------- n_consequents -------->|
        |     |-------||------|------|------|-----|------|
    n_rules   | rule0 || r0c0 | r0c1 | r0c2 | ... | r0cJ |
        |     | rule1 || r1c0 | r1c1 | r1c2 | ... | r1cJ |
        v     | ruleM || rMc0 | rMc1 | rMc2 | ... | rMcJ |

    A single element e.g. r1c2 is either a class in [0, n_classes-1] if the
    the problem is a classification problem, or a real value if the problem
    is a regression problem.
    All columns/consequents are NOT mutually exclusive. Therefore, for iris
    classification problem, you surely want to have 1 consequent that can take
    the values {0, 1, 2} mapped to "virginica", "setosa", "versicolor" since
    they ARE mutually exclusive.
    You can even have a mixed problem (classification and regression). For
    example, just add a column/consequent to the previous iris example. Let's
    call this new consequent the pollen concentration (i.e. a real value of an
    arbitrary unit. Note: here the value is always a positive
    number but negative numbers are supported too.

    """

    def __init__(
        self,
        X_train: np.array,
        y_train: np.array,
        # problem_type: ProblemType,
        n_rules: int,
        n_classes_per_cons: List[int],
        default_cons: np.array,
        n_max_vars_per_rule: int,
        n_labels_per_mf: int,
        n_labels_per_cons: LabelEnum = Label3,
        p_positions_per_lv: int = 32,  # 5 bits
        dc_weight: int = 1,
        mfs_shape: MFShape = MFShape.TRI_MF,
        # p_positions_per_cons: int = None,#32,  # 5 bits
        n_lv_per_ind_sp1: int = None,
    ):
        """

        :param X_train:
        :param y_train:
        :param n_rules:
        :param n_classes_per_cons: [n_classes_cons0, n_classes_cons1, ...]
        where n_class_consX is the number of classes for the X-th consequent.
        If the consequent is a continuous variable (i.e. regression) set the
        value to 0.
        :param n_max_vars_per_rule:
        :param n_labels_per_mf:
        :param p_positions_per_lv: Integer to represent the
        number of p positions (i.e. the possible values the membership functions
        (MFs) of a linguistic variable (LV) can take). For example, if
        p_positions_per_lv=4, then a MF's inflexion points will be at 0%, 33%,
        66% 100% of the variable range. The linguistic variable will be cut in
        p_positions_per_lv. This value must be a multiple of 2.
        :param dc_weight: integer. If =1 and
        n_labels_per_mf=3 -> [0,1,2,3] where 3 is DONT_CARE, =2 -> [0,1,2,3,
        4] where 3 and 4 is DONT_CARE, ... TODO: improve this comment
        :param mfs_shape:
        :param n_lv_per_ind_sp1: Integer to represent the number of MF encoded
        per individual of sp1. Must be >= n_max_vars_per_rule, ideally a
        multiple of 2. If not will be ceil-ed to the closest multiple of 2. If
        the problem you try to solve is big you maybe should increase this
        number
        """

        super().__init__()
        self._X, norm_args = minmax_norm(
            X_train
        )  # TODO: norm_args are used to de-normalized X
        self._y = y_train
        self._n_rules = n_rules
        self._n_classes_per_cons = np.asarray(n_classes_per_cons)
        self._default_cons = np.asarray(default_cons)
        self._n_max_vars_per_rule = n_max_vars_per_rule
        self._n_true_labels = n_labels_per_mf
        self._n_labels_cons = n_labels_per_cons
        self._p_positions_per_lv = p_positions_per_lv
        self._dc_weight = dc_weight
        self._mfs_shape = mfs_shape

        self._n_vars = self._X.shape[1]

        if n_lv_per_ind_sp1 is None:
            n_lv_per_ind_sp1 = self._n_max_vars_per_rule

        self._n_bits_per_lv = ceil(log(n_lv_per_ind_sp1, 2))
        print("n bits per lv", self._n_bits_per_lv)

        try:
            self._n_cons = self._y.shape[1]
        except IndexError:  # y is 1d so each element is an output
            self._n_cons = 1  # self._y.shape[0]

        self._cons_n_labels = self._compute_cons_n_labels(self._n_classes_per_cons)
        self._validate()

        self._n_bits_per_mf = ceil(log(self._p_positions_per_lv, 2))
        self._n_bits_per_ant = ceil(log(self._n_vars, 2))
        self._n_bits_per_cons = self._compute_n_bits_per_cons()
        print("bits per cons ", self._n_bits_per_cons)
        # chosen arbitrarily # ceil(log(self._n_true_labels + self._dc_padding, 2))
        self._n_bits_per_label = 5

        self._n_bits_sp1 = self._compute_needed_bits_for_sp1()
        self._n_bits_sp2 = self._compute_needed_bits_for_sp2()

        # contains True if i-th cons is a classification variable or False if regression
        self._cons_type = [bool(c) for c in self._n_classes_per_cons]

        self._cons_range = self._compute_cons_range()
        print("cons ra")
        print(self._cons_range)

        # self._nce = NativeCocoEvaluator(
        #
        # )
        self._nce = NativeCocoEvaluator(
            X_train=self._X,
            n_vars=self._n_vars,
            n_rules=self._n_rules,
            n_max_vars_per_rule=self._n_max_vars_per_rule,
            n_bits_per_mf=self._n_bits_per_mf,
            n_true_labels=self._n_true_labels,
            n_bits_per_lv=self._n_bits_per_lv,
            n_bits_per_ant=self._n_bits_per_ant,
            n_cons=self._n_cons,
            n_bits_per_cons=self._n_bits_per_cons,
            n_bits_per_label=self._n_bits_per_label,
            dc_weight=dc_weight,
            cons_n_labels=self._cons_n_labels,
            default_cons=self._default_cons,
        )

    def _validate(self):
        assert self._dc_weight >= 0, "negative padding does not make sense"
        assert log(self._p_positions_per_lv, 2) == ceil(
            log(self._p_positions_per_lv, 2)
        ), "p_positions_per_lv must be a multiple of 2"

        assert self._p_positions_per_lv >= self._n_true_labels, (
            "You must have at least as many p_positions as the n_labels_per_mf "
            "you want to use "
        )

        # if self._problem_type == ProblemType.CLASSIFICATION:
        #     msg = "You must have at least as many p_positions as the " \
        #           "n_classes you want to target "
        #     max_n_classes = self._get_highest_n_classes_per_cons()
        #     self._p_positions_per_cons =
        #     assert self._p_positions_per_cons >= max_n_classes, msg
        # elif self._problem_type == ProblemType.REGRESSION:

        ## Validate the number of classes per consequent
        n_classes_per_cons_in_y = np.apply_along_axis(
            lambda c: len(np.unique(c)), arr=self._y, axis=0
        ).reshape(-1)
        # force to have an array with a shape

        msg = (
            "the number of consequents indicated in n_class_per_cons does not "
            "match what was computed on y_train. from user: {}, computed: {}"
        )
        print(self._n_classes_per_cons)
        print(n_classes_per_cons_in_y)

        assert len(self._n_classes_per_cons) == len(n_classes_per_cons_in_y)

        n_cls_per_cons_zeroed = n_classes_per_cons_in_y.copy()
        # we don't want to compare the number of classes for continuous vars
        n_cls_per_cons_zeroed[self._n_classes_per_cons == 0] = 0
        assert np.array_equal(
            self._n_classes_per_cons.flatten(), n_cls_per_cons_zeroed.flatten()
        ), msg.format(self._n_classes_per_cons, n_classes_per_cons_in_y)

        assert all(
            [c >= 0 for c in self._n_classes_per_cons]
        ), "n_classes values must be positive in n_classes_per_cons"

        mask = n_classes_per_cons_in_y == self._n_classes_per_cons
        print("n cls per cons", self._n_classes_per_cons)
        print("mask", mask)
        assert all(
            mask[self._n_classes_per_cons != 0]
        ), "the n_classes per consequent does not match with what found on X_train"

        assert (
            2 ** self._n_bits_per_lv >= self._n_max_vars_per_rule
        ), "n_lv_per_ind_sp1 must be at least equals to n_max_vars_per_rule"

        # assert (
        #     self._n_labels_cons >= 2
        # ), "n_labels_cons must be >= 2 (i.e. at least LOW and HIGH)"

        assert issubclass(
            self._n_labels_cons, LabelEnum
        ), "n_labels _cons must an instance of a subclass of LabelEnum"

        assert self._default_cons.shape[0] == self._n_cons, (
            "default_cons's shape doesn't match the number of "
            "consequents retrieved using y_train"
        )

        # we check if a cons is either an int or the same class as
        # self._n_labels_cons. For the latter we check like that instead of
        # check issubclass because we do care that the label values of both
        # n_labels_cons and default_cons are the same (e.g. if
        # n_labels_cons's LOW = 0, then default_cons' LOW = 0 too)
        are_all_labels_or_int = all(
            [isinstance(c, (int, self._n_labels_cons)) for c in self._default_cons]
        )

        assert are_all_labels_or_int, (
            "The default rule must only contain classes or labels"
            " i.e. integer numbers. If a label is provide like LabelX.LOW"
            " make sure that the X in LabelX is the same for both n_labels_cons"
            " and default_cons"
        )
        #
        # # validate that the n classes per cons matches len(LabelX) used
        # # in the default rule cons. Valid example: n_classes_per_cons=3,
        # # default_cons = [Label3.LOW]. Invalid example: n_classes_per_cons=3,
        # # default_cons = [Label6.LOW] because 6!=3.
        # default_cons_filtered = [c for c in self._default_cons if isinstance(c, LabelEnum)]
        # assert [for (cons, n_labels) in zip(self._default_cons, self._cons_n_labels)]
        #
        #
        # # def valid(cons, n_labels):
        #
        #
        # convert LabelEnum to int
        self._default_cons = [
            x if isinstance(x, int) else x.value for x in self._default_cons
        ]

        def can_default_cons_fit_in_cons():
            for a, b in zip(self._default_cons, self._cons_n_labels):
                print("a ", a)
                print("b ", b)
                try:
                    yield a.value < b
                except AttributeError:
                    yield a < b

        print("lalalala", self._cons_n_labels)

        # assert (self._default_cons < self._cons_n_labels).all(), (
        assert all(can_default_cons_fit_in_cons()), (
            "Make sure that the default rule contains valid classes/labels \n"
            "i.e. label is in [0, n_classes-1] or in case of regression in \n"
            "[0, n_labels-1].\n"
            "Expected: ({}) < {}".format(self._default_cons, self._cons_n_labels)
        )

    def convert_to_fis(self, pyf_file):
        pass

    # def convert_to_fis(self, ind):
    #     pass

    def get_y_true(self):
        return self._y

    # def predict(self, ind):
    #     pass
    def predict(self, ind_tuple, X=None):
        """

        :param ind_tuple:
        :param X: if None will use self._X_train
        :return:
        """
        if X is not None:
            if not self._is_new_X_valid(X):
                raise ValueError(
                    "X parameter is not valid. Please check if it "
                    "is similar to X_train i.e. the X np.array "
                    "that was used to build the model"
                )
            X_minmax_normed = minmax_norm(X)
            y_pred = self._nce.predict(ind_sp1.to01(), ind_sp2.to01(), X_minmax_normed)
        ind_sp1, ind_sp2 = ind_tuple
        # pass ind_sp{1,2} in string format to make it easy to use it C++
        y_pred = self._nce.predict_native(ind_sp1.to01(), ind_sp2.to01())
        return y_pred

    def generate_sp1(self):
        return self._generate_ind(self._n_bits_sp1)

    def generate_sp2(self):
        return self._generate_ind(self._n_bits_sp2)

    @staticmethod
    def _generate_ind(n_bits):
        bin_str = "".join([str(randint(0, 1)) for _ in range(n_bits)])
        return bitarray(bin_str)

    def _compute_needed_bits_for_sp1(self):
        n_lv_per_ind = 2 ** self._n_bits_per_lv
        return int(n_lv_per_ind * self._n_true_labels * self._n_bits_per_mf)

    def _compute_needed_bits_for_sp2(self):
        # bits for r_sel_vars
        n_bits_r_sel_vars = (
            self._n_rules * self._n_max_vars_per_rule * self._n_bits_per_ant
        )

        # bits for r_lv
        n_bits_r_lv = self._n_rules * self._n_max_vars_per_rule * self._n_bits_per_lv

        # bits for r_labels
        n_bits_r_labels = (
            self._n_rules * self._n_max_vars_per_rule * self._n_bits_per_label
        )

        # # TODO depends if classif (scale to [0, n_class] ->
        # # ceil(log(n_class)) + modulo trick or regress (scale to [min(cons_i),
        # # max(cons_i)])
        #
        # # if classif:
        # #     n_bits = ceil(log(n_classes, 2))  # + modulo trick
        # # else:  # regression
        # #     n_bits = ceil(log(n_bits_per_cons, 2))
        #
        # assert p_positions_per_cons >= n_classes or problem is not classif  # n_classes = np.unique(y)
        # # this will return a BIG number for regression problem which will
        # # pass the assertion. NOPE! this is the exact opposite!
        # n_bits = ceil(log(p_positions_per_cons, 2))

        # TODO check if we need an other matrix to store the LV for cons. I
        # don't think so, because since n_cons << n_vars we can store all the
        # consequents in the individual's genome.

        # bits for r_cons
        n_bits_r_cons = self._n_rules * self._n_cons * self._n_bits_per_cons

        n_total_bits = n_bits_r_sel_vars + n_bits_r_lv + n_bits_r_labels + n_bits_r_cons

        for v in (n_bits_r_sel_vars, n_bits_r_lv, n_bits_r_labels, n_bits_r_cons):
            print("v", v)

        print("n_total_bits", n_total_bits)
        return int(n_total_bits)

    @staticmethod
    def clone(ind: bitarray):
        super().clone(ind)
        return ind.copy()

    # def _get_highest_n_classes_per_cons(self):
    #     n_classes_per_cons = np.apply_along_axis(
    #         lambda c: len(np.unique(c)), arr=self._X, axis=0
    #     )
    #     return np.max(n_classes_per_cons)

    def _compute_n_bits_per_cons(self):
        print("n class er cons", self._n_classes_per_cons)
        n_max_classes = max(self._n_classes_per_cons)

        # if all consequents are continuous variables (i.e. regression
        # i.e. value = 0) then we use a minimum of self._n_labels_per_cons)
        n_max_classes = max(n_max_classes, self._n_labels_cons.len())
        return ceil(log(n_max_classes, 2))

    def _compute_cons_range(self):
        """
        :return: a numpy array like:
            [[min_cons0, max_cons0],
             [min_cons1, max_cons1],
             [min_cons2, max_cons2],
             [....]]
        """
        return np.vstack([self._y.min(axis=0), self._y.max(axis=0)]).T

    def _compute_cons_n_labels(self, n_classes_per_cons):
        cons_n_labels = n_classes_per_cons.copy().astype(np.int)
        cons_n_labels[cons_n_labels == 0] = self._n_labels_cons.len()
        return cons_n_labels
