#    This file is part of DEAP.
#
#    DEAP is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as
#    published by the Free Software Foundation, either version 3 of
#    the License, or (at your option) any later version.
#
#    DEAP is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public
#    License along with DEAP. If not, see <http://www.gnu.org/licenses/>.
import math
import random

import numpy
from deap import algorithms
from deap import base
from deap import creator
from deap import tools

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()


def randfloat(_min, _max):
    return random.random() * _max + _min


# Attribute generator
toolbox.register("attr_speed", randfloat, 0, 15)
toolbox.register("attr_angle", randfloat, 0, 90)

# Structure initializers
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_speed, toolbox.attr_angle), 2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def distance_range(speed, angle):
    return (speed * speed) / 9.81 * math.sin(2.0 * math.radians(angle))


def eval_fitness(individual):
    speed = extract_speed_from_ind(individual)
    angle = extract_angle_from_ind(individual)
    return distance_range(speed, angle),


def extract_angle_from_ind(individual):
    return individual[1] + individual[3]


def extract_speed_from_ind(individual):
    return individual[0] + individual[2]


def describe_individual(individual):
    print("-------------------------------------------------------------------")
    print("individual: {}".format(individual))
    print("speed: {:.2f} m/s".format(extract_speed_from_ind(individual)))
    print("angle: {:.2f}°".format(extract_angle_from_ind(individual)))
    print("-> distance: {:.2f} m".format(eval_fitness(individual)[0]))
    print("-------------------------------------------------------------------")


toolbox.register("evaluate", eval_fitness)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)


def main():
    # random.seed(64)

    pop = toolbox.population(n=200)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=500,
                                   stats=stats, halloffame=hof, verbose=True)

    best_inds = tools.selBest(pop, k=2)
    for ind in best_inds:
        # print(ind)
        describe_individual(ind)

    print("theoretical reference")
    describe_individual(individual=[15, 22.5, 15, 22.5])

    return pop, log, hof


if __name__ == "__main__":
    main()
