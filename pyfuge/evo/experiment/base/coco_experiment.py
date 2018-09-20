import random
from collections import namedtuple
from functools import partial
from itertools import product
from typing import Callable

import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap.algorithms import varAnd
from deap.tools import HallOfFame

from pyfuge.evo.experiment.base.experiment import Experiment
from pyfuge.evo.experiment.coco.coco_individual import CocoIndividual
from pyfuge.evo.helpers import fis_individual

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

SpeciesTuple = namedtuple('SpeciesTuple', ['sp1', 'sp2', 'fitness'])

species_indices_dict = {
    "sp1": [0, 1],
    "sp2": [1, 0]
}


class CocoExperiment(Experiment):
    """
    A class that performs an experiment using a cooperative-coevolution
    algorithm with DEAP library.
    """

    def __init__(self, coco_ind: CocoIndividual, fitness_func: Callable):
        super().__init__(fitness_func)
        self._coco_ind = coco_ind
        self._post_init()

    def run(self):
        self.coop()

    def coop(self):
        verbose = True

        # Cooperative-Coevolution parameters
        # INDIVIDUAL_LENGTH = 200
        POP_SIZE = 100
        N_GEN = 300
        N_REPRESENTATIVES = 5
        CX_PROB = 0.9
        MUT_PROB = 0.5
        N_ELITE = 3
        N_HOF = 2

        # Problem specific parameters
        # MIN_SPEED, MAX_SPEED = 0, 10
        # MIN_ANGLE, MAX_ANGLE = 20, 180

        # Setup logbook
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        logbook = tools.Logbook()
        logbook.header = "gen", "species", "std", "min", "avg", "max"

        # Setup individuals

        # if the _coco_ind implements its own clone method, then we use it
        # otherwise let's use the DEAP/default one
        if isinstance(self._coco_ind, fis_individual.Clonable):
            toolbox.register("clone", self._coco_ind.clone)

        # Setup sp1 species
        creator.create("IndividualSp1", list, fitness=creator.FitnessMax)
        toolbox.register("species_sp1", tools.initRepeat, list,
                         self._coco_ind.generate_sp1)

        # Setup angle species
        creator.create("IndividualSp2", list, fitness=creator.FitnessMax)
        toolbox.register("species_sp2", tools.initRepeat, list,
                         self._coco_ind.generate_sp2)

        def eval_solution(species_indices, ind_tuple):
            ind_sp1 = ind_tuple[species_indices[0]]
            ind_sp2 = ind_tuple[species_indices[1]]
            y_pred = self._coco_ind.predict(ind_sp1, ind_sp2)
            y_true = self._coco_ind.get_y_true()

            fitness = self._fitness_func(y_true, y_pred)
            return fitness,  # DEAP expects a tuple for fitnesses
            # TODO: self._fitness_func(y_true, y_pred, ind=(ind_sp1, ind_sp2)
            #  gen=g, pop=pop)

        # def eval_solution(species_indices, ind_tuple):
        #     ind_speed = ind_tuple[species_indices[0]]
        #     ind_angle = ind_tuple[species_indices[1]]
        #
        #     speed = extract_speed_from_ind(ind_speed)
        #     angle = extract_angle_from_ind(ind_angle)
        #
        #     # validate domains for speed and angle
        #     assert MIN_SPEED <= speed < MAX_SPEED, "{}".format(ind_speed)
        #     assert MIN_ANGLE <= angle < MAX_ANGLE, "{}".format(ind_angle)
        #
        #     return distance_range(speed, angle),

        def select_representatives(individuals, k):
            k0 = k // 2
            k1 = k - k0
            representatives = []
            representatives.extend(tools.selBest(individuals, k0))
            representatives.extend(tools.selRandom(individuals, k1))
            return representatives

        # setup evo functions, they are common to both species
        toolbox.register("evaluate", eval_solution)
        toolbox.register("mate", tools.cxTwoPoint)
        # FIXME: mutFlipBit will produce invalid ind (in this example at least)
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.3)
        toolbox.register("select_elite", tools.selBest)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register("select_representatives", select_representatives,
                         k=N_REPRESENTATIVES)

        # create species_sp1 and species_sp2
        species_sp1 = toolbox.species_sp1(n=POP_SIZE)
        species_sp2 = toolbox.species_sp2(n=POP_SIZE)

        all_species = {
            "sp1": species_sp1,
            "sp2": species_sp2
        }

        # In the generation 0 the representatives (best individual used in
        # co-evolution) are chosen randomly
        representatives_sp1 = [random.choice(species_sp1) for _ in
                               range(N_REPRESENTATIVES)]
        representatives_sp2 = [random.choice(species_sp2) for _ in
                               range(N_REPRESENTATIVES)]

        # Hall Of Fame (hof) contains the best couples/pairs seen across all
        # generations. They will be the ones that we will keep when N_GEN is
        # reached.
        hof = HallOfFame(maxsize=N_HOF)

        print("Generation 0")
        evaluate_species(all_species["sp1"], "sp1", representatives_sp2,
                         N_REPRESENTATIVES, hof)
        evaluate_species(all_species["sp2"], "sp2", representatives_sp1,
                         N_REPRESENTATIVES, hof)

        for g in range(1, N_GEN + 1):
            representatives_sp1 = \
                evolve_species(all_species, "sp1", representatives_sp2,
                               CX_PROB,
                               MUT_PROB, N_REPRESENTATIVES, POP_SIZE, N_ELITE,
                               hof)
            update_logbook(all_species, "sp1", g, logbook, stats, verbose)

            representatives_sp2 = \
                evolve_species(all_species, "sp2", representatives_sp1,
                               CX_PROB,
                               MUT_PROB, N_REPRESENTATIVES, POP_SIZE, N_ELITE,
                               hof)
            update_logbook(all_species, "sp2", g, logbook, stats, verbose)

            # print("Best fitness generation {gen}: {fit}"
            #       .format(gen=g, fit=hof[0].fitness))

        couple = hof[0]
        print("ind with fitness {.3f}".format(couple.fitness))
        self._coco_ind.print_ind(couple.sp1, couple.sp2)

        # print("theoretical reference")
        # describe_individual(ind_speed=[MAX_SPEED, MAX_SPEED],
        #                     ind_angle=[45, 45],
        #                     fitness=[distance_range(MAX_SPEED, 45)])


def update_logbook(all_species, species_name, g, logbook, stats, verbose):
    record = stats.compile(all_species[species_name])
    logbook.record(gen=g, species=species_name, **record)

    if verbose:
        print(logbook.stream)


def evolve_species(all_species, species_name, other_species_representatives,
                   CX_PROB, MUT_PROB, N_REPRESENTATIVES, POP_SIZE, N_ELITE,
                   hof):
    species = all_species[species_name]

    # (1) Elite select
    elite = toolbox.select_elite(species, N_ELITE)

    # (2) Select
    offspring = toolbox.select(species, POP_SIZE - N_ELITE)

    # add elite to offspring
    offspring.extend(elite)

    # (3) Crossover and Mutate offspring
    offspring = varAnd(offspring, toolbox, CX_PROB, MUT_PROB)

    # (4) Evaluate the entire population
    evaluate_species(offspring, species_name, other_species_representatives,
                     N_REPRESENTATIVES, hof)

    # (5) Replace the current population by the offspring
    species[:] = offspring

    # (6) Select representatives.
    return toolbox.select_representatives(species)


def evaluate_species(species, species_name, other_species_representatives,
                     N_REPRESENTATIVES, hof):
    # reevaluate only the individual that have been mutated/crossed.
    # FIXME: is this still valid with coev?, because representative could have
    # changed invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    invalid_ind = species

    species_indices = species_indices_dict[species_name]
    partial_evaluate = partial(toolbox.evaluate, species_indices)
    fitnesses = list(map(partial_evaluate,
                         product(invalid_ind, other_species_representatives)))

    # Update Hall of Fame (i.e. the best individuals couples)
    update_hof(hof, invalid_ind, N_REPRESENTATIVES, fitnesses,
               other_species_representatives, species_indices)


def update_hof(hof, invalid_ind, N_REPRESENTATIVES, fitnesses,
               other_species_representatives, species_indices):
    for i, ind in enumerate(invalid_ind):
        idx_start = N_REPRESENTATIVES * i
        idx_end = idx_start + N_REPRESENTATIVES

        # take the max fitness for a given individual since it is evaluated
        # N times where N is the number of individuals
        idx_representative, fitness = max(
            enumerate(fitnesses[idx_start:idx_end]), key=lambda x: x[1])

        ind.fitness.values = fitness

        best_representative = other_species_representatives[idx_representative]

        t_ind = (ind, best_representative)
        hof.update((SpeciesTuple(
            sp1=t_ind[species_indices[0]],
            sp2=t_ind[species_indices[1]],
            fitness=fitness),
        ))
