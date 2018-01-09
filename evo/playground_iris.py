from string import ascii_lowercase

import numpy as np

target_solution = "deap library is cool"
target_length = len(target_solution)


def main():
    import random
    from deap import creator, base, tools, algorithms

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    possible_solutions = ascii_lowercase + " _.,"
    toolbox.register("attr_bool", random.choice, possible_solutions)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_bool, n=target_length)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_one_max(individual):
        # print("".join(individual), target_solution)
        # a = [1.0 / (1.0 + abs(ord(t_c) - ord(c))) for c, t_c in
        #      zip("".join(individual), target_solution)]

        a = np.mean([1.0 / (1.0 + abs(ord(t_c) - ord(c))) for c, t_c in
                     zip("".join(individual), target_solution)]),

        # print(a)
        return a
        # for c, t_c in zip("".join(individual), target_solution):
        #     if ord(t_c) == ord(c):
        #         n_true += 1
        #
        # return n_true,

    def yolo(individual, indpb):
        # FIXME: wrong doctring
        """Flip the value of the attributes of the input individual and return the
        mutant. The *individual* is expected to be a :term:`sequence` and the values of the
        attributes shall stay valid after the ``not`` operator is called on them.
        The *indpb* argument is the probability of each attribute to be
        flipped. This mutation is usually applied on boolean individuals.

        :param individual: Individual to be mutated.
        :param indpb: Independent probability for each attribute to be flipped.
        :returns: A tuple of one individual.

        This function uses the :func:`~random.random` function from the python base
        :mod:`random` module.
        """
        for i in range(len(individual)):
            if random.random() < indpb:
                individual[i] = random.choice(possible_solutions)

        return individual,

    toolbox.register("evaluate", eval_one_max)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", yolo, indpb=0.10)
    toolbox.register("select", tools.selTournament, tournsize=3)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    population = toolbox.population(n=300)

    NGEN = 100

    hof = tools.HallOfFame(1)

    """
    HOF == kind of elitism : "[...] use of a HallOfFame in order to keep track of the best individual to appear in the evolution (it keeps it even in the case it extinguishes),"
    HOF: max = 1 if any of the fitness tuple is equal to 1
    example:
        - (0.5, 1.0, 0.33) --> 1.0
        - np.mean((0.5, 1.0, 0.33)) --> 0.XX --> 0.XX
        
    ERRATA: to use a fitness with mulitple values, use:
    creator.create("FitnessMulti", base.Fitness, weights=(-1.0, 1.0))
    """

    algorithms.eaSimple(population, toolbox, cxpb=0.5, mutpb=0.1, ngen=NGEN,
                        halloffame=hof,
                        stats=stats)

    # for gen in range(NGEN):
    #     offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
    #     fits = toolbox.map(toolbox.evaluate, offspring)
    #     for fit, ind in zip(fits, offspring):
    #         ind.fitness.values = fit
    #     population = toolbox.select(offspring, k=len(population))
    top10 = tools.selBest(population, k=50)

    for t in top10:
        print("".join(t))


if __name__ == '__main__':
    from time import time

    t0 = time()
    tick = lambda: print((time() - t0) * 1000)

    main()
    tick()
