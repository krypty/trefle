from enum import Enum
from math import ceil, log
from random import randint
from typing import List, Type

import numpy as np
from bitarray import bitarray
from sklearn.preprocessing import MinMaxScaler

from trefle.evo.experiment.coco.coco_individual_validator import \
    CocoIndividualValidator
from trefle.evo.experiment.coco.fixed_size_bitarray_factory import (
    FixedSizeBitArrayFactory,
)
from trefle.evo.experiment.coco.native_coco_evaluator import NativeCocoEvaluator
from trefle.evo.helpers.fis_individual import FISIndividual, Clonable
from trefle.evo.helpers.fuzzy_labels import LabelEnum, Label3

"""
Convention: a variable to represent the number of bits needed for a matrix
must start with n_bits_XXX.
It is defined like: n_bits_XXX = n_rows * n_cols * n_bits_per_element
"""


class MFShape(Enum):
    TRI_MF = 0
    TRAP_MF = 1


class CocoIndividual(FISIndividual, Clonable):
    """
    This class creates two individuals i.e. one to represent the membership
    functions and the other to represent the rules, called respectively
    specie 1 (sp1) and specie 2 (sp2). Each of these species is basically
    an 1d array of bits. This array is then split into n matrices (where n
    differs from sp1 and sp2) which are used to build the fuzzy system.

    Reference: Carlos Pena's thesis Table 3.1 chapter Application Example The
    Iris Problem although this implementation is an exact reproduction of what
    is described there.

    In short the process to transform these 1d arrays of bits to a fuzzy system
    is. This process is done in the C++ part in fis_coco_eval_wrapper.cpp's
    extract_XXX() methods. Here are the main steps of this process:
      1. Extract from the 1d array of bits n matrices
         "110001111011000110"  ->  [110 001  and   [ 000
                                    111 011]         110 ]
      2. Convert the bit matrices into "business" matrices i.e. matrices
      with numbers that represents something for the fuzzy system (e.g. the
      index of the variable to used for a particular antecedent.
         [110 001  and   [ 000     -> [6  1    and [0
          111 011]         110 ]       7  3]        6]
      For example this could means that the 2nd rule uses the var 7 and 3 and
      set the consequent to 6th class.

      3. Uses these matrices to build the fuzzy system and all their parts (
      fuzzy rules, linguistic variables,...)


    **Specie 1 - the MFs**
    The shape of the 1d bit array of sp1 is the following:
        [ lv0p0, lv0p1, ..., lv0pK, lv1p0, lv1p2, ..., lv1pK,
          lv2p0, lv2p1, ..., lvIp0, lvIp1, ...., ...,  lvIpK ]

    Where:
        K = p_position_per_lv
        I = n_lv_per_ind_sp1

    An individual of sp1 (i.e. an instance of this bit array) can been seen as
    a pool of linguistic variables defined by their respective p-points. The
    size of the pool is I.

    This bit array will be converted to a matrix called r_mfs.

    r_mfs is a KxI matrix like this:

        ^                ||<------------- p_positions --------->|
        |          |-----||-------|-------|-------|-----|-------|
    n_lv_per_ind   | lv0 || lv0p0 | lv0p1 | lv0p2 | ... | lv0pK |
        |          | lv1 || lv1p0 | lv1p1 | lv1p2 | ... | lv1pK |
        v          | lvI || lvIp0 | lvIp1 | lvIp2 | ... | lvIpK |


    Meaning: lvipj is the j-th point (or p position) of the i-th linguistic
    variable of the pool. lvipj values are in [0, 1]. This kind of represents the
    percentage of a variable (which needs to be scaled in [0,1] beforehand).
    You might want to pre-process outliers before running TrefleClassifier.
    For example, for a triangular LV (triLV) the value lv4p1 is represented by
    'p1' on the x-axis:


        Membership functions for variable 4
      ^
      | low      medium           high
    1 |    X       X          XX
      |     X     X  X       XX
      |      X   X    X    XX
      |       X X      XX X
      |       XX        XXX
      |      X  X     XX   XX
      |     X    X XX       XX
      |    X       X          XX
    0 +--------------------------------------> 1
           p0     *p1*        p2

    In "lvipK" the "K" defines the number of p-positions a linguistic variable has.
    In the case of a triangular membership function (as it has been implemented
    so far), the number of p-positions (i.e. K) is >= n_labels (i.e. the
    linguistic labels like LOW, MEDIUM and HIGH).

    The resolution (i.e. the number of possible values that a p position can
    take is defined by the p_positions_per_lv. So the number of bits needed to
    represent the desired p_positions_per_lv is:
    n_bits_per_mf = ceil(log2(p_positions_per_lv))

    The total number of bits used is given by _compute_needed_bits_for_sp1()


    **Specie 2 - the rules**



    **THIS IS OUTDATED** !!!
    r_sel_vars
    r_lv
    r_labels
    r_cons


    The shape of the 1d bit array of sp2 is the following:
        [ r0a0,   r0a1, ... r0aN,   r1a0,   r1a1, ... rMaN,
          r0lv0,  r0lv1,... r0lvN,  r1lv0,  r1lv1,... rMlvN,
          r0lbl0, r0lbl1,...r0lblN, r1lbl0, r1lbl1,...rMlblN,
          r0c0, r0c1,...r0cJ, r1c0, r1c1,...rMcJ ]

    Where:
        M = n_rules
        N = n_max_vars_per_rule
        J = n_consequents i.e. number of output variables that are NOT mutually
            exclusive

    The rules are composed of two parts. The antecedents (Aij) and the
    consequents (Cij).

    The Aij part indicates which antecedents/variables are used by the
    model/fis for all its rules. It consists of three "matrices" let's call
    them r_sel_vars, r_lv and r_labels.

    r_sel_vars is a NxM matrix like this:

        ^             ||<----- n_max_vars_per_rule ----->|
        |     |-------||------|------|------|-----|------|
    n_rules   | rule0 || r0a0 | r0a1 | r0a2 | ... | r0aN |
        |     | rule1 || r1a0 | r1a1 | r1a2 | ... | r1aN |
        v     | ruleM || rMa0 | rMa1 | rMa2 | ... | rMaN |

    A single element e.g. r1a2 is a link/"pointer" to the variable index/number
    to use for the 3rd antecedent (a2) of the 2nd rule (r1). A single element
    is in the interval [0, n_vars-1] where n_vars is deduced from X_train.


    r_lv is a NxM matrix like this:

        ^             ||<------- n_max_vars_per_rule ------->|
        |     |-------||-------|-------|-------|-----|-------|
    n_rules   | rule0 || r0lv0 | r0lv1 | r0lv2 | ... | r0lvN |
        |     | rule1 || r1lv0 | r1lv1 | r1lv2 | ... | r1lvN |
        v     | ruleM || rMlv0 | rMlv1 | rMlv2 | ... | rMlvN |

    A single element e.g. r0lv2 is a link/"pointer" to a linguistic variable
    defined in an individual from sp1. This element's value is a number
    representing the index of the lv (i.e. a number in [0, I-1] where I
    is n_lv_per_ind) to use from the paired ind_sp1. For example if r0lv2=4 then
    the linguistic variable used for the variable of the 1st rule (r0) and the 3rd (lv2)
    antecedent will be the 5th row of the matrix r_mfs of the paired ind_sp1.
    Note: since elements of r_lv (i.e. antecedents) are pointers to a
    linguistic variable (lvI) it is possible that:
       1. 2 or more antecedents points to the same lvI (i.e. surjectivity)
       2. Not all lvI are pointed by an element (i.e. injectivity)
       3. 2 antecedents or more use the same variable but point to different lvI

    The point 3. is a problem because the interpretability criteria tell us that
    a fuzzy variable should use the same definition across all rules. To fix this
    problem, we only keep the last definition of a variable. This is done in
    the C++ part with the map vars_lv_lookup.

    r_labels is a NxM matrix like this:

        ^             ||<--------- n_max_vars_per_rule --------->|
        |     |-------||--------|--------|--------|-----|--------|
    n_rules   | rule0 || r0lbl0 | r0lbl1 | r0lbl2 | ... | r0lblN |
        |     | rule1 || r1lbl0 | r1lbl1 | r1lbl2 | ... | r1lblN |
        v     | ruleM || rMlbl0 | rMlbl1 | rMlbl2 | ... | rMlblN |

    A single element e.g. r1mf2 is a link/"pointer" to a linguistic label
    (by label we mean for example 0 for LOW, 1 for MEDIUM, 2 for HIGH, ...
    last for DONT_CARE) to use for the variable/antecedent r1a2 of the
    r_sel_vars "matrix". For example if r1a2=6, r1lbl2=2 and n_true_labels=4
    (i.e. VERY LOW, LOW, MEDIUM, HIGH) then it would mean that the 3rd
    antecedent (a2) of the 2nd rule (r1) is "5th variable is MEDIUM". (5th
    because of "=6" and MEDIUM because of lbl=2 with n_true_labels=4). A single
    element is in the interval [0, (n_true_labels + dc_weight)-1].
    With dc_weight=1 all labels have the same probability to be chosen.
    To increase the probability to chose/select a DC, increase dc_weight.


    The other part of sp2 is Cij. It defines the consequents (i.e. classes
    in a classification problem or a label in a regression problem) that are
    used by the model/fis for all its rules. It consists of a single "matrix"
    let's call it r_cons.

    r_cons is a JxM matrix like:


        ^             ||<-------- n_consequents -------->|
        |     |-------||------|------|------|-----|------|
    n_rules   | rule0 || r0c0 | r0c1 | r0c2 | ... | r0cJ |
        |     | rule1 || r1c0 | r1c1 | r1c2 | ... | r1cJ |
        v     | ruleM || rMc0 | rMc1 | rMc2 | ... | rMcJ |

    A single element e.g. r1c2 is either a class in [0, n_classes-1] if the
    the problem is a classification problem, or a label if the problem
    is a regression problem.
    All columns/consequents are NOT mutually exclusive. Therefore, for iris
    classification problem, you surely want to have 1 consequent that can take
    the values {0, 1, 2} mapped to "virginica", "setosa", "versicolor" since
    they ARE mutually exclusive.
    Therefore, if you want to transform this iris problem into a one-hot version
    of it, create 3 consequents with 2 classes per consequent.
    You can even have a mixed problem (classification and regression). For
    example, just add a column/consequent to the previous iris example. Let's
    call this new consequent the pollen concentration (i.e. a label, let's say
    LOW. This label is a positive integer in [0, n_labels_per_cons-1].
    """

    def __init__(
        self,
        X_train: np.array,
        y_train: np.array,
        n_rules: int,
        n_classes_per_cons: List[int],
        default_cons: np.array,
        n_max_vars_per_rule: int,
        n_labels_per_mf: int,
        n_labels_per_cons: Type[LabelEnum] = Label3,
        p_positions_per_lv: int = 32,  # 5 bits
        dc_weight: int = 1,
        n_lv_per_ind_sp1: int = None,
    ):
        """

        :param X_train: a 2d numpy array representing the train data. Each
        column is a variable and each row is an observation.

        :param y_train: a 2d numpy array representing the output classes/values
        for the train data. Each column is a mutually exclusive output (either
        a class or a regression value) and each row is the outputs for an
        observation.

        :param n_rules: the number of rules the fuzzy system have. The total
        number of rules is n_rules + 1 (the default rule).

        :param n_classes_per_cons: [n_classes_cons0, n_classes_cons1, ...]
        where n_class_consX is the number of classes for the X-th consequent.
        If the consequent is a continuous variable (i.e. regression) set the
        value to 0.

        :param default_cons: array of numbers to set the default consequent(s)
        for the default rule. For a consequent representing a class, specify the
        class directly. For a consequent representing a continuous variable,
        specify LabelXXX.YYY (where XXX is the same number as n_labels_per_cons,
        e.g. Label3 and YYY is a label value of this class, e.g. Label3.LOW()).
        Example: for a problem with 2 consequents where the 1st represents a
        class (e.g. n_classes=6) and the 2nd is a continuous variable (e.g.
        split with n_labels_per_cons=Label4) then you can set
        default_cons=[3, Label4.HIGH()]

        :param n_max_vars_per_rule: Maximum number of variables to use for a
        single rule. Use this parameter to reduce the size of the fuzzy system.

        :param n_labels_per_mf: number of labels per membership function. For
        example, n_labels_per_mf=4 will correspond to "low, medium, high,
        very high"

        :param n_labels_per_mf: number of labels per membership function. For
        example, n_labels_per_mf=4 will correspond to "low, medium, high,
        very high"

        :param n_labels_per_cons: number of labels per membership function for
        the consequents (singleton MFs). For example, use 4 to have LOW, MEDIUM,
        HIGH, VERY_HIGH labels. These labels are equally spaced in the range
        (i.e. definition domain of y_train) so when n_labels_per_cons=4, LOW is
        set to 1/4 of abs(cons_min - cons_max).

        :param p_positions_per_lv: Integer to represent the
        number of p positions (i.e. the possible values the membership functions
        (MFs) of a linguistic variable (LV) can take). For example, if
        p_positions_per_lv=4, then a MF's inflexion points will be at 0%, 33%,
        66% 100% of the variable range. In others words, the linguistic variable
        will be cut in p_positions_per_lv. This value must be a multiple of 2.

        :param dc_weight: integer. Set the don't care weight. If dc_weight=k
        then a variable v has k more chance to be a don't care. Setting
        dc_weight=0 will lead to create rules that have exactly
        n_max_vars_per_rule. Setting dc_weight to a big number will to lead
        less rules than n_rules because all their antecedents will be set to
        don't care.

        :param n_lv_per_ind_sp1: This is an advanced parameter. it's an integer
        to represent the number of MF encoded per individual of sp1. In other
        words it is the pool of MF where the linguistic variable from ind_sp2
        will be defined from. Must be >= n_max_vars_per_rule, ideally a
        multiple of 2. If not will be ceil-ed to the closest multiple of 2. If
        the problem you try to solve is big you maybe should increase this
        number. By default, this value is set (before ceiling) to
        n_max_vars_per_rule * n_rules. This ensures that each LV can have its
        own MF.
        """

        super().__init__()
        self._X, self._X_scaler = CocoIndividual._minmax_norm(X_train)
        self._y = y_train
        self._n_rules = n_rules
        self._n_classes_per_cons = np.asarray(n_classes_per_cons)
        self._default_cons = np.asarray(default_cons)
        self._n_max_vars_per_rule = n_max_vars_per_rule
        self._n_true_labels = n_labels_per_mf
        self._n_labels_cons = n_labels_per_cons
        self._p_positions_per_lv = p_positions_per_lv
        self._dc_weight = dc_weight
        self._mfs_shape = MFShape.TRI_MF

        self._n_vars = self._X.shape[1]

        if self._n_max_vars_per_rule is None:
            self._n_max_vars_per_rule = self._n_vars

        if n_lv_per_ind_sp1 is None:
            n_lv_per_ind_sp1 = self._n_max_vars_per_rule * self._n_rules

        self._n_bits_per_lv = ceil(log(n_lv_per_ind_sp1, 2))

        try:
            self._n_cons = self._y.shape[1]
        except IndexError:  # y is 1d so each element is an output
            self._n_cons = 1

        self._cons_n_labels = self._compute_cons_n_labels(self._n_classes_per_cons)

        CocoIndividualValidator(self).validate()
        self._default_cons = self._convert_labelenum_to_int(self._default_cons)

        self._n_bits_per_mf = ceil(log(self._p_positions_per_lv, 2))
        self._n_bits_per_ant = ceil(log(self._n_vars, 2))
        self._n_bits_per_cons = self._compute_n_bits_per_cons()

        # chosen arbitrarily, enough to cover a high number of labels (i.e. 2**5=32)
        self._n_bits_per_label = 5

        self._n_bits_sp1 = self._compute_needed_bits_for_sp1()
        self._n_bits_sp2 = self._compute_needed_bits_for_sp2()
        self._ind_sp1_class = FixedSizeBitArrayFactory.create(self._n_bits_sp1)
        self._ind_sp2_class = FixedSizeBitArrayFactory.create(self._n_bits_sp2)

        # contains True if i-th cons is a classification variable or False if regression
        self._cons_type = [bool(c) for c in self._n_classes_per_cons]

        self._cons_scaler = self._create_cons_scaler()

        self._cons_range = np.vstack(
            (self._cons_scaler.data_min_, self._cons_scaler.data_max_)
        ).T.astype(np.double)

        self._vars_range = self._create_vars_range(self._X_scaler)

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
            n_classes_per_cons=self._n_classes_per_cons,
            default_cons=self._default_cons,
            vars_range=self._vars_range,
            cons_range=self._cons_range,
        )

    def predict(self, ind_tuple, X=None):
        ind_sp1, ind_sp2 = self._extract_ind_tuple(ind_tuple)

        if X is None:
            y_pred = self._nce.predict_native(ind_sp1, ind_sp2)
        else:
            X_normed = self._X_scaler.transform(X)
            y_pred = self._nce.predict_native(ind_sp1, ind_sp2, X_normed)

        return self._post_predict(y_pred)

    def to_tff(self, ind_tuple):
        ind_sp1, ind_sp2 = self._extract_ind_tuple(ind_tuple)
        return self._nce.to_tff(ind_sp1, ind_sp2)

    def get_y_true(self):
        return self._y

    def print_ind(self, ind_tuple):
        ind_sp1, ind_sp2 = self._extract_ind_tuple(ind_tuple)
        self._nce.print_ind(ind_sp1, ind_sp2)

    def get_ind_sp1_class(self):
        return self._ind_sp1_class

    def get_ind_sp2_class(self):
        return self._ind_sp2_class

    @staticmethod
    def clone(ind: bitarray):
        return ind.deep_copy()

    def _create_cons_scaler(self):
        # y_pred returned by NativeCocoEvaluator are in range
        # [0, n_class_per_cons-1] and it needs to be scaled back to
        # [min_val_cons, max_val_cons] (which for binary and multiclass
        # consequents do nothing but this is needed for continuous variables)

        cons_scaler = MinMaxScaler()
        cons_scaler.fit(self._y.astype(np.double))
        return cons_scaler

    @staticmethod
    def _extract_ind_tuple(ind_tuple):
        # convert ind_sp{1,2} in string format to make it easy to use it C++
        return ind_tuple[0].bits.to01(), ind_tuple[1].bits.to01()

    def _post_predict(self, y_pred):
        return self._scale_back_y(y_pred)

    @staticmethod
    def _generate_ind(n_bits):
        bin_str = format(randint(0, (2 ** n_bits) - 1), "0{}b".format(n_bits))
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

        # bits for r_cons
        n_bits_r_cons = self._n_rules * self._n_cons * self._n_bits_per_cons

        n_total_bits = n_bits_r_sel_vars + n_bits_r_lv + n_bits_r_labels + n_bits_r_cons
        return int(n_total_bits)  # int cast because of the multiple ceil() used above

    def _compute_n_bits_per_cons(self):
        n_max_classes = max(self._n_classes_per_cons)

        # if all consequents are continuous variables (i.e. regression
        # i.e. value = 0) then we use a minimum of self._n_labels_per_cons)
        n_max_classes = max(n_max_classes, self._n_labels_cons.len())
        return ceil(log(n_max_classes, 2))

    def _compute_cons_n_labels(self, n_classes_per_cons):
        cons_n_labels = n_classes_per_cons.copy().astype(np.int)
        cons_n_labels[cons_n_labels == 0] = self._n_labels_cons.len()
        return cons_n_labels

    def _scale_back_y(self, y):
        # -1 because y is in [0, cons_n_labels-1]
        y_ = y / (self._cons_n_labels - 1)
        return self._cons_scaler.inverse_transform(y_)

    @staticmethod
    def _minmax_norm(X_train):
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        return X_train_scaled, scaler

    @staticmethod
    def _create_vars_range(scaler):
        vars_range = np.vstack((scaler.data_min_, scaler.data_max_)).T.astype(np.double)
        return vars_range

    @staticmethod
    def _convert_labelenum_to_int(labels_enums):
        def is_int_or_numpy_int(v):
            try:
                return issubclass(v.dtype.type, (int, np.integer))
            except:
                return False

        return [
            cons if is_int_or_numpy_int(cons) else cons.value for cons in labels_enums
        ]
