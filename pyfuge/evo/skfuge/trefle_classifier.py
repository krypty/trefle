from math import ceil
from typing import Type

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import mean_squared_error

from pyfuge.evo.experiment.base.coco_experiment import CocoExperiment
from pyfuge.evo.experiment.coco.coco_individual import CocoIndividual
from pyfuge.evo.experiment.view.coco_evolution_viewer import CocoEvolutionViewer
from pyfuge.evo.helpers.fuzzy_labels import LabelEnum, Label3
from pyfuge.trefle.tffconverter import TffConverter

PERCENTAGE_ELITE = 0.1
PERCENTAGE_HOF = 0.3


class TrefleClassifier(BaseEstimator, ClassifierMixin):
    _fitness_function_signature = "def fit(y_true, y_pred): return fitness"

    @staticmethod
    def _default_fit_func(y_true, y_pred):
        return -mean_squared_error(y_true, y_pred)

    def __init__(
        self,
        # fuzzy systems parameters
        n_rules,
        n_classes_per_cons,
        default_cons,
        n_max_vars_per_rule,
        n_labels_per_mf,
        n_labels_per_cons: Type[LabelEnum] = Label3,
        p_positions_per_lv: int = 32,
        dc_weight: int = 1,
        n_lv_per_ind_sp1=None,
        # cooperative-coevolution parameters
        pop_size=80,
        n_generations=100,
        n_representatives=4,
        crossover_prob=0.5,
        # crossover_prob=0,
        mutation_prob=0.03,
        # mutation_prob=1,
        n_elite=None,
        halloffame_size=None,
        fitness_function=None,
        # other parameters
        verbose=False,
    ):
        """

        :param n_rules:
        :param n_labels_per_mf:
        :param pop_size:
        :param n_generations:
        :param halloffame:
        :param fitness_function: a function like fitness(y_true, y_pred) that
        returns a scalar value. The higher this value is the better the fitness.
        Be careful to use classification metrics on a classification and vice
        versa for regression problem unless you know what you do (e.g. use RMSE
        on a classification problem)
        :param verbose:
        """

        self.n_rules = n_rules
        self.n_classes_per_cons = n_classes_per_cons
        self.default_cons = default_cons
        self.n_max_vars_per_rule = n_max_vars_per_rule
        self.n_labels_per_mf = n_labels_per_mf
        self.n_labels_per_cons = n_labels_per_cons
        self.p_positions_per_lv = p_positions_per_lv
        self.dc_weight = dc_weight
        self.n_lv_per_ind_sp1 = n_lv_per_ind_sp1

        self.pop_size = pop_size
        self.n_generations = n_generations
        self.fitness_function = fitness_function
        self.n_representatives = n_representatives
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.n_elite = n_elite
        self.halloffame_size = halloffame_size

        self.verbose = verbose

    def _validate(self):
        if self.n_labels_per_mf <= 1:
            raise ValueError("n_labels_per_mf must be > 1")

        if not isinstance(self.dc_weight, int) or self.dc_weight < 0:
            raise ValueError("dc_weight must be positive integer")

        if self.fitness_function is None:
            self.fitness_function = self._default_fit_func
        elif not hasattr(self.fitness_function, "__call__"):
            raise ValueError(
                "fitness function must be a callable like: {}".format(
                    self._fitness_function_signature
                )
            )

        if self.n_elite is None:
            self.n_elite = int(ceil(PERCENTAGE_ELITE * self.pop_size))

        if self.halloffame_size is None:
            self.halloffame_size = int(ceil(PERCENTAGE_HOF * self.pop_size))

    def fit(self, X, y):
        self._validate()

        self._fis_ind = CocoIndividual(
            X_train=X,
            y_train=y,
            n_rules=self.n_rules,
            n_classes_per_cons=self.n_classes_per_cons,
            default_cons=self.default_cons,
            n_max_vars_per_rule=self.n_max_vars_per_rule,
            n_labels_per_mf=self.n_labels_per_mf,
            n_labels_per_cons=self.n_labels_per_cons,
            p_positions_per_lv=self.p_positions_per_lv,
            dc_weight=self.dc_weight,
            n_lv_per_ind_sp1=self.n_lv_per_ind_sp1,
        )

        exp = CocoExperiment(
            coco_ind=self._fis_ind,
            n_generations=self.n_generations,
            pop_size=self.pop_size,
            n_representatives=self.n_representatives,
            crossover_prob=self.crossover_prob,
            mutation_prob=self.mutation_prob,
            n_elite=self.n_elite,
            halloffame_size=self.halloffame_size,
            fitness_func=self.fitness_function,
        )
        exp.run()

        if self.verbose:
            logbook = exp.get_logbook()
            CocoEvolutionViewer.plot_fitness(logbook)

        self._best_ind = exp.get_top_n()[0]
        return self

    def predict(self, X):
        self._ensure_fit()

        # var_ranges = IndEvaluatorUtils.compute_vars_range(X)

        # ind_evaluator = NativeIndEvaluator(
        #     ind_n=len(self._best_ind),
        #     observations=X,
        #     n_rules=self.n_rules,
        #     max_vars_per_rule=self._n_vars,  # TODO remove me
        #     n_labels=len(self._mf_label_names),
        #     n_consequents=len(self._default_rule_output),
        #     default_rule_cons=np.array(self._default_rule_output),
        #     vars_ranges=var_ranges,
        #     labels_weights=self._labels_weights,
        # )

        ind_tuple = (self._best_ind.sp1, self._best_ind.sp2)
        y_pred = self._fis_ind.predict(ind_tuple, X)
        # return self._compute_y_pred_bin(y_pred)
        return y_pred

    # def get_last_unthresholded_predictions(self):
    #     self._ensure_fit()
    #
    #     return self._y_pred

    def get_best_fuzzy_system(self):
        self._ensure_fit()

        best_ind = self._best_ind
        ind_tuple = (best_ind.sp1, best_ind.sp2)

        self._fis_ind.print_ind(ind_tuple)

        tff_fis = self._fis_ind.to_tff(ind_tuple)
        # TODO: here convert to SingletonFIS or TrefleFIS using
        # TffConverter
        # TffConverter.to_fis(tff_fis)
        return tff_fis

    def _ensure_fit(self):
        if getattr(self, "_fis_ind") is None:
            raise ValueError("You must use fit() first")

    @staticmethod
    def _compute_y_pred_bin(y_pred):
        if y_pred.shape[1] > 1:
            # return the class with the highest output
            return np.argmax(y_pred, axis=1)
        else:
            # output is binary
            return np.round(y_pred)
