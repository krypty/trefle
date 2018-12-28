from math import ceil
from typing import Type, List, Callable

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import mean_squared_error

from trefle.evo.experiment.base.coco_experiment import CocoExperiment
from trefle.evo.experiment.coco.coco_individual import CocoIndividual
from trefle.evo.helpers.fuzzy_labels import LabelEnum, Label3
from trefle.fitness_functions.output_thresholder import round_to_cls


class TrefleClassifier(BaseEstimator, ClassifierMixin):
    _fitness_function_signature = "def fit(y_true, y_pred): return fitness"
    _PERCENTAGE_ELITE = 0.1
    _PERCENTAGE_HOF = 0.3

    @staticmethod
    def _default_fit_func(y_true, y_pred):
        return -mean_squared_error(y_true, y_pred)

    def __init__(
        self,
        # fuzzy systems parameters
        n_rules: int,
        n_classes_per_cons: List[int],
        default_cons: List[int],
        n_max_vars_per_rule: int,
        n_labels_per_mf: int,
        n_labels_per_cons: Type[LabelEnum] = Label3,
        p_positions_per_lv: int = 32,
        dc_weight: int = 1,
        n_lv_per_ind_sp1: int = None,
        # cooperative-coevolution parameters
        pop_size: int = 80,
        n_generations: int = 100,
        n_representatives: int = 4,
        crossover_prob: float = 0.5,
        mutation_prob: float = 0.03,
        n_elite: int = None,
        halloffame_size: int = None,
        fitness_function: Callable = None,
        # other parameters
        verbose: bool = False,
    ):
        """

        The documentation for the fuzzy systems parameters is in CocoIndividual
        class and for the evolutionary parameters it is in CocoExperiment class

        :param default_cons:
        :param n_max_vars_per_rule:
        :param n_labels_per_mf:
        :param n_labels_per_cons:
        :param p_positions_per_lv:
        :param dc_weight:
        :param n_lv_per_ind_sp1:
        :param pop_size:
        :param n_generations:
        :param n_representatives:
        :param crossover_prob:
        :param mutation_prob:
        :param n_elite:
        :param halloffame_size:
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
            self.n_elite = int(ceil(TrefleClassifier._PERCENTAGE_ELITE * self.pop_size))

        if self.halloffame_size is None:
            self.halloffame_size = int(
                ceil(TrefleClassifier._PERCENTAGE_HOF * self.pop_size)
            )

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

        self._experiment = CocoExperiment(
            coco_ind=self._fis_ind,
            n_generations=self.n_generations,
            pop_size=self.pop_size,
            n_representatives=self.n_representatives,
            crossover_prob=self.crossover_prob,
            mutation_prob=self.mutation_prob,
            n_elite=self.n_elite,
            halloffame_size=self.halloffame_size,
            fitness_func=self.fitness_function,
            verbose=self.verbose,
        )
        self._experiment.run()

        self._best_ind = self._experiment.get_top_n()[0]
        return self

    def predict(self, X):
        self._ensure_fit()

        ind_tuple = (self._best_ind.sp1, self._best_ind.sp2)
        y_pred = self._fis_ind.predict(ind_tuple, X)
        return y_pred

    def predict_classes(self, X):
        y_pred = self.predict(X)

        for i, n_classes in enumerate(self.n_classes_per_cons):
            if n_classes > 0:  # not a continuous variable
                y_pred[:, i] = round_to_cls(y_pred[:, i], n_classes)
        return y_pred

    def get_best_fuzzy_system_as_tff(self):
        self._ensure_fit()

        ind_tuple = self._get_ind_tuple(self._best_ind)

        tff_str = self._fis_ind.to_tff(ind_tuple)
        return tff_str

    def print_best_fuzzy_system(self):
        self._ensure_fit()
        ind_tuple = self._get_ind_tuple(self._best_ind)
        self._fis_ind.print_ind(ind_tuple)

    @staticmethod
    def _get_ind_tuple(ind):
        ind_tuple = (ind.sp1, ind.sp2)
        return ind_tuple

    def get_experiment_logbook(self):
        self._ensure_fit()
        return self._experiment.get_logbook()

    def _ensure_fit(self):
        if getattr(self, "_fis_ind") is None:
            raise ValueError("You must use fit() first")
