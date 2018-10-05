import random
from collections import Counter
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
from pyfuge.evo.experiment.coco.coco_couple import CocoCouple
from pyfuge.evo.experiment.coco.coco_individual import CocoIndividual
from pyfuge.evo.helpers import fis_individual

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

species_indices_dict = {"sp1": [0, 1], "sp2": [1, 0]}


class CocoExperiment(Experiment):
    """
    A class that performs an experiment using a cooperative-coevolution
    algorithm with DEAP library.
    """

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
    ):
        super().__init__(fitness_func)
        self._coco_ind = coco_ind
        self._n_generations = n_generations
        self._pop_size = pop_size
        self._n_representatives = n_representatives
        self._crossover_prob = crossover_prob
        self._mutation_prob = mutation_prob
        self._n_elite = n_elite
        self._halloffame_size = halloffame_size
        self._post_init()

    def run(self):
        self.coop()
        self.plot_fitness()

    def plot_fitness(self):
        import matplotlib.pyplot as plt

        gen = self._logbook.select("gen")
        gen = list(set(gen))
        fit = self._logbook.select("species", "max")
        max_fit_sp1 = [v[1] for v in zip(*fit) if v[0] == "sp1"]
        max_fit_sp2 = [v[1] for v in zip(*fit) if v[0] == "sp2"]

        avg_hof = self._logbook.select("avg_hof")
        # drop half the points to have the same as the number of generations
        avg_hof = [v for i, v in enumerate(avg_hof) if i % 2 == 0]

        fig, ax1 = plt.subplots()
        line1 = ax1.plot(gen, max_fit_sp1, "b-", label="Max fit sp1")
        ax1.set_xlabel("Generation")
        ax1.set_ylabel("Fitness", color="b")
        for tl in ax1.get_yticklabels():
            tl.set_color("b")

        ax2 = ax1.twinx()

        line2 = ax2.plot(gen, avg_hof, "r-", label="Max fit sp2")
        ax2.set_ylabel("Fitness sp2", color="r")
        for tl in ax2.get_yticklabels():
            tl.set_color("r")

        lns = line1 + line2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc="center right")

        plt.show()

    def coop(self):
        verbose = True

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

        # Setup sp1 species
        creator.create(
            "IndividualSp1",
            self._coco_ind.get_ind_sp1_class(),
            fitness=creator.FitnessMax,
        )
        toolbox.register("species_sp1", tools.initRepeat, list, creator.IndividualSp1)

        # Setup angle species
        creator.create(
            "IndividualSp2",
            self._coco_ind.get_ind_sp2_class(),
            fitness=creator.FitnessMax,
        )
        toolbox.register("species_sp2", tools.initRepeat, list, creator.IndividualSp2)

        def eval_solution(species_indices, ind_tuple):
            ind_sp1 = ind_tuple[species_indices[0]]
            ind_sp2 = ind_tuple[species_indices[1]]
            y_pred = self._coco_ind.predict((ind_sp1, ind_sp2))
            y_true = self._coco_ind.get_y_true()

            fitness = self._fitness_func(y_true, y_pred)
            # print(ind_sp1)
            # print("ind1 {}, ind2 {} fit: {:.3f}".format(ind_sp1.bits.to01(), ind_sp2.bits.to01(), fitness))
            # print("hash ind1 {}, ind2 {} fit: {:.3f}".format(hash(ind_sp1.bits.to01()), hash(ind_sp2.bits.to01()), fitness))
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

        def yolo_twoPoints(ind1, ind2):

            """Executes a two-point crossover on the input :term:`sequence`
            individuals. The two individuals are modified in place and both keep
            their original length.

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
                ind2[cxpoint1:cxpoint2].true_deep_copy(),
                ind1[cxpoint1:cxpoint2].true_deep_copy(),
            )

            return ind1, ind2

        toolbox.register("mate", yolo_twoPoints)
        # toolbox.register("mate", tools.cxTwoPoint)
        # FIXME: mutFlipBit will produce invalid ind (in this example at least)
        # Will have to define our own mutate flip bit if we use bitarray

        def flip_bit(individual, indpb):
            for i in range(len(individual.bits) - 1):
                if random.random() < indpb:
                    individual.bits[i : i + 1] = ~individual.bits[i : i + 1]
            return (individual,)

        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=MUT_PROB)
        # toolbox.register("mutate", flip_bit, indpb=0.3)
        toolbox.register("select_elite", tools.selBest)
        toolbox.register("select", tools.selTournament, tournsize=3)
        toolbox.register(
            "select_representatives", select_representatives, k=N_REPRESENTATIVES
        )

        # create species_sp1 and species_sp2
        species_sp1 = toolbox.species_sp1(n=POP_SIZE)
        species_sp2 = toolbox.species_sp2(n=POP_SIZE)

        all_species = {"sp1": species_sp1, "sp2": species_sp2}

        # ind_1 = species_sp1[0]
        # ind_1.fitness = (0.123,)
        # ind_2 = species_sp1[1]

        # print("A", ind_1.bits, ind_1.fitness)
        # ind_1_copy = ind_1.true_deep_copy()
        # ind_1_mutated, = toolbox.mutate(ind_1_copy)
        # # pop = [ind_1]
        # # offspring = [toolbox.clone(i) for i in pop]
        # # offspring = varAnd(pop, toolbox, cxpb=0, mutpb=1)
        # print("a", ind_1.bits, ind_1.fitness)
        # print("b", ind_1_mutated.bits, ind_1_mutated.fitness)
        # print("c", ind_2.bits, ind_2.fitness)
        #
        # assert False, "yololol"

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

        print("Generation 0")
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
                MUT_PROB,
                N_REPRESENTATIVES,
                POP_SIZE,
                N_ELITE,
                self._hof,
            )
            update_logbook(
                self._hof, all_species, "sp1", g, self._logbook, stats, verbose
            )

            representatives_sp2 = evolve_species(
                all_species,
                "sp2",
                representatives_sp1,
                CX_PROB,
                MUT_PROB,
                N_REPRESENTATIVES,
                POP_SIZE,
                N_ELITE,
                self._hof,
            )
            update_logbook(
                self._hof, all_species, "sp2", g, self._logbook, stats, verbose
            )

            # print("Best fitness generation {gen}: {fit}"
            #       .format(gen=g, fit=hof[0].fitness))

        couple = self._hof[0]
        print("ind with fitness {:.3f}".format(couple.fitness[0]))
        self._coco_ind.print_ind((couple.sp1, couple.sp2))

        # print("theoretical reference")
        # describe_individual(ind_speed=[MAX_SPEED, MAX_SPEED],
        #                     ind_angle=[45, 45],
        #                     fitness=[distance_range(MAX_SPEED, 45)])

    def get_top_n(self):
        return self._hof


def update_logbook(hof, all_species, species_name, g, logbook, stats, verbose):
    avg_hof = np.mean([pair.fitness for pair in hof.items])
    record = stats.compile(all_species[species_name])
    logbook.record(avg_hof=avg_hof, gen=g, species=species_name, **record)

    if verbose:
        print("--------", ([h for h in hof]))
        print(logbook.stream)


def evolve_species(
    all_species,
    species_name,
    other_species_representatives,
    CX_PROB,
    MUT_PROB,
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
    offspring = varAnd(offspring, toolbox, CX_PROB, MUT_PROB)
    # offspring = varAnd(offspring, toolbox, CX_PROB, mutpb=1)

    # add elite to offspring
    offspring.extend(elite)

    # (4) Evaluate the entire population
    evaluate_species(
        offspring, species_name, other_species_representatives, N_REPRESENTATIVES, hof
    )

    yolo = [hash(off.bits.to01()) for off in offspring]
    c = Counter(yolo)
    print(c.most_common())

    # (5) Replace the current population by the offspring
    species[:] = offspring

    # (6) Select representatives.
    return toolbox.select_representatives(species)


def evaluate_species(
    species, species_name, other_species_representatives, n_representatives, hof
):
    # reevaluate only the individual that have been mutated/crossed.
    # FIXME: is this still valid with coev?, because representative could have
    # changed invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    invalid_ind = species

    species_indices = species_indices_dict[species_name]
    partial_evaluate = partial(toolbox.evaluate, species_indices)
    fitnesses = list(
        map(partial_evaluate, product(invalid_ind, other_species_representatives))
    )
    # print("fitness len", len(fitnesses))

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
    invalid_ind,
    n_representatives,
    fitnesses,
    other_species_representatives,
    species_indices,
):
    for i, ind in enumerate(invalid_ind):
        idx_start = n_representatives * i
        idx_end = idx_start + n_representatives

        # take the max fitness for a given individual since it is evaluated
        # N times where N is the number of individuals
        idx_representative, fitness = max(
            enumerate(fitnesses[idx_start:idx_end]), key=lambda x: x[1]
        )

        fitnesses_values = [f[0] for f in fitnesses[idx_start:idx_end]]
        # mean_fitness_ind = np.median(fitnesses_values)
        mean_fitness_ind = max(fitnesses_values)
        # mean_fitness_ind = sum(fitnesses_values)/ (idx_end-idx_start)

        ind.fitness.values = (mean_fitness_ind,)

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
