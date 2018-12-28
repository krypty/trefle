import random
import warnings
from functools import partial
from itertools import product
from typing import Callable

import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap.algorithms import varAnd
from deap.tools import HallOfFame

from trefle.evo.experiment.base.experiment import Experiment
from trefle.evo.experiment.coco.coco_couple import CocoCouple
from trefle.evo.experiment.coco.coco_individual import CocoIndividual
from trefle.evo.helpers import fis_individual

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

species_indices_dict = {"sp1": [0, 1], "sp2": [1, 0]}


class CocoExperiment(Experiment):
    """
    A class that performs an experiment using a cooperative-coevolution
    algorithm with DEAP library. The logic of cooperative-coevolution is handled
    by this class and it is does not need to know how the individuals are
    encoded nor how they are combined to make predictions.
    The individuals' encoding and evaluation/prediction are done by a
    CocoIndividual object.
    """

    def get_logbook(self):
        return self._logbook

    def __init__(
        self,
        coco_ind: CocoIndividual,
        n_generations: int,
        pop_size: int,
        n_representatives: int,
        crossover_prob: float,
        mutation_prob: float,
        n_elite: int,
        halloffame_size: int,
        fitness_func: Callable,
        verbose: bool,
    ):
        """

        :param coco_ind: an instance of CocoIndividual that knows how the
        individuals are encoded and evaluated.
        :param n_generations: number of generations to run the evolution
        :param pop_size: population size for specie 1 and specie 2
        :param n_representatives: number of representatives (i.e. individuals)
        to select to mate with the other specie. The representatives are
        composed of randomly selected individuals and of the best ones.
        :param crossover_prob: crossover probability for specie 1 and specie 2
        :param mutation_prob: mutation probability for specie 1 and specie 2.
        Every single individual in the population can mutate
        (i.e. inter-individual mutation probability=1) but the individual itself
        (i.e. intra-individual mutation probability or "the bits inside the
        individual") will mutate with a probability of `mutation_prob`.
        :param n_elite: keep the `n_elite` best individuals between generation
        g and g+1 to avoid losing good individuals. This is done separately
        for both specie 1 and specie 2.
        :param halloffame_size: keep `halloffame_size` couples (i.e. an
        individual from specie 1 and one for specie 2) that have the highest
        fitness across all generations. This is what is left/returned after the
        evolution process. The top 1 couple from the hall of fame is the couple
        with the highest fitness over all generations.
        :param fitness_func: a callable with the signature `f(y_true, y_pred)`
        that return a number. The higher this number is the higher the fitness
        value is.
        :param verbose: if True a log with statistics about the populations
        will be printed out.
        """
        super().__init__(fitness_func)
        self._coco_ind = coco_ind
        self._n_generations = n_generations
        self._pop_size = pop_size
        self._n_representatives = n_representatives
        self._crossover_prob = crossover_prob
        self._mutation_prob = mutation_prob
        self._n_elite = n_elite
        self._halloffame_size = halloffame_size
        self._verbose = verbose
        self._post_init()

    def run(self):
        self.coop()

    def coop(self):
        # Cooperative-Coevolution parameters
        N_GEN = self._n_generations
        POP_SIZE = self._pop_size
        N_REPRESENTATIVES = self._n_representatives
        CX_PROB = self._crossover_prob
        MUT_PROB = self._mutation_prob
        N_ELITE = self._n_elite
        HOF_SIZE = self._halloffame_size

        # Setup logbook
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        self._logbook = tools.Logbook()
        self._logbook.header = "gen", "species", "std", "min", "avg", "max", "avg_hof"

        # Setup individuals

        # if the _coco_ind implements its own clone method, then we use it
        # otherwise let's use the DEAP/default one
        if isinstance(self._coco_ind, fis_individual.Clonable):
            toolbox.register("clone", self._coco_ind.clone)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            # Setup sp1 species
            creator.create(
                "IndividualSp1",
                self._coco_ind.get_ind_sp1_class(),
                fitness=creator.FitnessMax,
            )
            toolbox.register(
                "species_sp1", tools.initRepeat, list, creator.IndividualSp1
            )

            # Setup sp2 species
            creator.create(
                "IndividualSp2",
                self._coco_ind.get_ind_sp2_class(),
                fitness=creator.FitnessMax,
            )
            toolbox.register(
                "species_sp2", tools.initRepeat, list, creator.IndividualSp2
            )

        def eval_solution(species_indices, ind_tuple):
            ind_sp1 = ind_tuple[species_indices[0]]
            ind_sp2 = ind_tuple[species_indices[1]]
            y_pred = self._coco_ind.predict((ind_sp1, ind_sp2))
            y_true = self._coco_ind.get_y_true()

            fitness = self._fitness_func(y_true, y_pred)
            return (fitness,)  # DEAP expects a tuple for fitnesses
            # TODO: self._fitness_func(y_true, y_pred, ind=(ind_sp1, ind_sp2)
            #  gen=g, pop=pop)

        def select_representatives(individuals, k):
            k0 = k // 2
            k1 = k - k0
            representatives = []
            representatives.extend(tools.selBest(individuals, k0))
            representatives.extend(tools.selRandom(individuals, k1))
            return representatives

        # setup evo functions, they are common to both species
        toolbox.register("evaluate", eval_solution)

        def two_point_crossover(ind1, ind2):
            """Executes a two-point crossover on the input :term:`sequence`
            individuals. The two individuals are modified in place and both keep
            their original length.
            The reason we do not use tools.cxTwoPoint is because we must
            make sure that a deep copy is performed while manipulating the
            individuals.

            :param ind1: The first individual participating in the crossover.
            :param ind2: The second individual participating in the crossover.
            :returns: A tuple of two individuals.

            This function uses the :func:`~random.randint` function from the Python
            base :mod:`random` module.
            """
            size = min(len(ind1), len(ind2))
            cxpoint1 = random.randint(1, size)
            cxpoint2 = random.randint(1, size - 1)
            if cxpoint2 >= cxpoint1:
                cxpoint2 += 1
            else:  # Swap the two cx points
                cxpoint1, cxpoint2 = cxpoint2, cxpoint1

            ind1[cxpoint1:cxpoint2], ind2[cxpoint1:cxpoint2] = (
                ind2[cxpoint1:cxpoint2].deep_copy(),
                ind1[cxpoint1:cxpoint2].deep_copy(),
            )

            return ind1, ind2

        toolbox.register("mate", two_point_crossover)
        # toolbox.register("mate", tools.cxTwoPoint)
        # FIXME: mutFlipBit will produce invalid ind (in this example at least)
        # Will have to define our own mutate flip bit if we use bitarray

        def flip_bit(individual, indpb):
            for i in range(len(individual.bits) - 1):
                if random.random() < indpb:
                    individual.bits[i : i + 1] = ~individual.bits[i : i + 1]
            return (individual,)

        # toolbox.register("mutate", tools.mutShuffleIndexes, indpb=MUT_PROB)
        toolbox.register("mutate", flip_bit, indpb=MUT_PROB)
        toolbox.register("select_elite", tools.selBest)
        toolbox.register("select", tools.selRoulette)
        toolbox.register(
            "select_representatives", select_representatives, k=N_REPRESENTATIVES
        )

        # create species_sp1 and species_sp2
        species_sp1 = toolbox.species_sp1(n=POP_SIZE)
        species_sp2 = toolbox.species_sp2(n=POP_SIZE)

        all_species = {"sp1": species_sp1, "sp2": species_sp2}

        # In the generation 0 the representatives (best individual used in
        # co-evolution) are chosen randomly
        representatives_sp1 = [
            random.choice(species_sp1) for _ in range(N_REPRESENTATIVES)
        ]
        representatives_sp2 = [
            random.choice(species_sp2) for _ in range(N_REPRESENTATIVES)
        ]

        # Hall Of Fame (hof) contains the best couples/pairs seen across all
        # generations. They will be the ones that we will keep when N_GEN is
        # reached.
        self._hof = HallOfFame(maxsize=HOF_SIZE)

        evaluate_species(
            all_species["sp1"], "sp1", representatives_sp2, N_REPRESENTATIVES, self._hof
        )
        evaluate_species(
            all_species["sp2"], "sp2", representatives_sp1, N_REPRESENTATIVES, self._hof
        )

        for g in range(1, N_GEN + 1):
            representatives_sp1 = evolve_species(
                all_species,
                "sp1",
                representatives_sp2,
                CX_PROB,
                N_REPRESENTATIVES,
                POP_SIZE,
                N_ELITE,
                self._hof,
            )
            self._update_logbook(self._hof, all_species, "sp1", g, self._logbook, stats)

            representatives_sp2 = evolve_species(
                all_species,
                "sp2",
                representatives_sp1,
                CX_PROB,
                N_REPRESENTATIVES,
                POP_SIZE,
                N_ELITE,
                self._hof,
            )
            self._update_logbook(self._hof, all_species, "sp2", g, self._logbook, stats)

    def get_top_n(self):
        return self._hof

    def _update_logbook(self, hof, all_species, species_name, g, logbook, stats):
        avg_hof = np.mean([pair.fitness for pair in hof.items])
        record = stats.compile(all_species[species_name])
        logbook.record(avg_hof=avg_hof, gen=g, species=species_name, **record)

        self._print_if_verbose(logbook.stream)

    def _print_if_verbose(self, text):
        if self._verbose:
            print(text)


def evolve_species(
    all_species,
    species_name,
    other_species_representatives,
    CX_PROB,
    N_REPRESENTATIVES,
    POP_SIZE,
    N_ELITE,
    hof,
):
    species = all_species[species_name]

    # (1) Elite select
    elite = toolbox.select_elite(species, N_ELITE)

    # (2) Select
    offspring = toolbox.select(species, POP_SIZE - N_ELITE)

    # (3) Crossover and Mutate offspring
    offspring = varAnd(offspring, toolbox, CX_PROB, mutpb=1.0)

    # add elite to offspring
    offspring.extend(elite)

    # (4) Evaluate the entire population
    evaluate_species(
        offspring, species_name, other_species_representatives, N_REPRESENTATIVES, hof
    )

    # (5) Replace the current population by the offspring
    species[:] = offspring

    # (6) Select representatives.
    return toolbox.select_representatives(species)


def evaluate_species(
    species, species_name, other_species_representatives, n_representatives, hof
):
    # consider all individuals as invalid and therefore ask for reevaluation
    invalid_ind = species

    species_indices = species_indices_dict[species_name]
    partial_evaluate = partial(toolbox.evaluate, species_indices)
    fitnesses = list(
        map(partial_evaluate, product(invalid_ind, other_species_representatives))
    )

    # Update Hall of Fame (i.e. the best individuals couples)
    update_hof(
        hof,
        invalid_ind,
        n_representatives,
        fitnesses,
        other_species_representatives,
        species_indices,
    )


def update_hof(
    hof,
    individuals,
    n_representatives,
    fitnesses,
    other_species_representatives,
    species_indices,
):
    """
    Compute the resulting fitness of all individuals given their fitnesses
    with all the other species representatives. This is done using the max()
    function.

    Then the best representative, i.e. the one that gave the maximum fitness
    is retrieved and the couple ind_spX-representative_spY is saved in the hall
    of fame.
    """
    for i, ind in enumerate(individuals):
        idx_start = n_representatives * i
        idx_end = idx_start + n_representatives

        # take the max fitness for a given individual since it is evaluated
        # N times where N is the number of individuals
        idx_representative, fitness = max(
            enumerate(fitnesses[idx_start:idx_end]), key=lambda x: x[1]
        )

        ind.fitness.values = fitness

        best_representative = other_species_representatives[idx_representative]

        t_ind = (ind, best_representative)
        hof.update(
            (
                CocoCouple(
                    sp1=t_ind[species_indices[0]],
                    sp2=t_ind[species_indices[1]],
                    fitness=fitness,
                ),
            )
        )
