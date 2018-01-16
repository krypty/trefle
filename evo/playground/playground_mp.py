def create_dataset():
    a = None
    with open("/home/gary/Downloads/win7office.ova", mode="rb") as f:
        a = f.read()

    print(len(a))
    return a


super_heavy_dataset = create_dataset()


def eval_ind(individual):
    a = len(super_heavy_dataset)
    kaka = super_heavy_dataset[20]
    x = 1000
    while x < 100000:
        x += 1
        y = x * x
    return sum(individual),  # this must be a tuple


def main():
    import random
    from deap import creator, base, tools, algorithms

    import multiprocessing

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    pool = multiprocessing.Pool()
    toolbox.register("map", pool.map)

    toolbox.register("attr_bool", random.randrange, 1, 100)
    toolbox.register("individual", tools.initRepeat, creator.Individual,
                     toolbox.attr_bool, n=100)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # def evalOneMax(individual):
    #     return eval_ind(individual)

    toolbox.register("evaluate", eval_ind)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=30)

    NGEN = 100
    for gen in range(NGEN):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.1)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))
    top10 = tools.selBest(population, k=10)

    for t in top10:
        print(t)


if __name__ == '__main__':
    main()
