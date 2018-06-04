import math
import random
from collections import namedtuple
from functools import partial
from itertools import product

import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap.algorithms import varAnd
from deap.tools import HallOfFame

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()


def distance_range(speed, angle):
    return (speed * speed) / 9.81 * math.sin(2.0 * math.radians(angle))


def eval_fitness(individual):
    speed = extract_speed_from_ind(individual)
    angle = extract_angle_from_ind(individual)
    return distance_range(speed, angle),


def extract_speed_from_ind(individual):
    return sum(individual) / len(individual)


def extract_angle_from_ind(individual):
    return sum(individual) / len(individual)


def describe_individual(ind_speed, ind_angle, fitness):
    print("-------------------------------------------------------------------")
    print("individual: {};{}".format(ind_speed, ind_angle))
    print("speed: {:.2f} m/s".format(extract_speed_from_ind(ind_speed)))
    print("angle: {:.2f}°".format(extract_angle_from_ind(ind_angle)))
    print("-> distance: {:.2f} m".format(fitness[0]))
    print("-------------------------------------------------------------------")


SpeciesTuple = namedtuple('SpeciesTuple', ['speed', 'angle', 'fitness'])


def coop():
    verbose = True

    # Cooperative-Coevolution parameters
    INDIVIDUAL_LENGTH = 200
    POP_SIZE = 100
    N_GEN = 300
    N_REPRESENTATIVES = 5
    CX_PROB = 0.9
    MUT_PROB = 0.5
    N_ELITE = 3
    N_HOF = 2

    # Problem specific parameters
    MIN_SPEED, MAX_SPEED = 0, 10
    MIN_ANGLE, MAX_ANGLE = 20, 180

    # Setup logbook
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = "gen", "species", "std", "min", "avg", "max"

    # Setup individuals
    # Setup speed species
    creator.create("IndividualSpeed", list, fitness=creator.FitnessMax)
    toolbox.register("attr_speed", random.uniform, MIN_SPEED,
                     MAX_SPEED)
    toolbox.register("individual_speed", tools.initRepeat,
                     creator.IndividualSpeed,
                     toolbox.attr_speed, INDIVIDUAL_LENGTH)
    toolbox.register("species_speed", tools.initRepeat, list,
                     toolbox.individual_speed)

    # Setup angle species
    creator.create("IndividualAngle", list, fitness=creator.FitnessMax)
    toolbox.register("attr_angle", random.uniform, MIN_ANGLE,
                     MAX_ANGLE)
    toolbox.register("individual_angle", tools.initRepeat,
                     creator.IndividualAngle,
                     toolbox.attr_angle, INDIVIDUAL_LENGTH)
    toolbox.register("species_angle", tools.initRepeat, list,
                     toolbox.individual_angle)

    def eval_solution(species_indices, ind_tuple):
        ind_speed = ind_tuple[species_indices[0]]
        ind_angle = ind_tuple[species_indices[1]]

        speed = extract_speed_from_ind(ind_speed)
        angle = extract_angle_from_ind(ind_angle)

        # validate domains for speed and angle
        assert MIN_SPEED <= speed < MAX_SPEED, "{}".format(ind_speed)
        assert MIN_ANGLE <= angle < MAX_ANGLE, "{}".format(ind_angle)

        return distance_range(speed, angle),

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

    # create species_speed and species_angle
    species_speed = toolbox.species_speed(n=POP_SIZE)
    species_angle = toolbox.species_angle(n=POP_SIZE)

    all_species = {
        "speed": species_speed,
        "angle": species_angle
    }

    # In the generation 0 the representatives (best individual used in
    # co-evolution) are chosen randomly
    representatives_speed = [random.choice(species_speed) for _ in
                             range(N_REPRESENTATIVES)]
    representatives_angle = [random.choice(species_angle) for _ in
                             range(N_REPRESENTATIVES)]

    # Hall Of Fame (hof) contains the best couples/pairs seen across all
    # generations. They will be the ones that we will keep when N_GEN is
    # reached.
    hof = HallOfFame(maxsize=N_HOF)

    print("Generation 0")
    evaluate_species(all_species["speed"], "speed", representatives_angle,
                     N_REPRESENTATIVES, hof)
    evaluate_species(all_species["angle"], "angle", representatives_speed,
                     N_REPRESENTATIVES, hof)

    for g in range(1, N_GEN + 1):
        representatives_speed = \
            evolve_species(all_species, "speed", representatives_angle, CX_PROB,
                           MUT_PROB, N_REPRESENTATIVES, POP_SIZE, N_ELITE, hof)
        update_logbook(all_species, "speed", g, logbook, stats, verbose)

        representatives_angle = \
            evolve_species(all_species, "angle", representatives_speed, CX_PROB,
                           MUT_PROB, N_REPRESENTATIVES, POP_SIZE, N_ELITE, hof)
        update_logbook(all_species, "angle", g, logbook, stats, verbose)

        # print("Best fitness generation {gen}: {fit}"
        #       .format(gen=g, fit=hof[0].fitness))

    couple = hof[0]
    describe_individual(couple.speed, couple.angle, couple.fitness)

    print("theoretical reference")
    describe_individual(ind_speed=[MAX_SPEED, MAX_SPEED],
                        ind_angle=[45, 45],
                        fitness=[distance_range(MAX_SPEED, 45)])


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

    species_indices = [0, 1] if species_name == "speed" else [1, 0]
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
            speed=t_ind[species_indices[0]],
            angle=t_ind[species_indices[1]],
            fitness=fitness),
        ))


if __name__ == "__main__":
    coop()
