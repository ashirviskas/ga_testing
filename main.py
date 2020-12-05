import numpy
import ga
from block import Block
from grid import Grid
import numpy as np
import random

"""
The y=target is to maximize this equation ASAP:
    y = w1x1+w2x2+w3x3+w4x4+w5x5+6wx6
    where (x1,x2,x3,x4,x5,x6)=(4,-2,3.5,5,-11,-4.7)
    What are the best values for the 6 weights w1 to w6?
    We are going to use the genetic algorithm for the best possible values after a number of generations.
"""


# Random number chance
def rnc(num, a, b, chance):
    if random.random() < chance:
        return random.randint(a, b)
    else:
        return num


def initialize_population(blocks, pop_size):
    population = np.tile(blocks, (pop_size, 1, 1))
    for i in range(1, len(population)):
        for j in range(len(population[i])):
            population[i, j, 0] = rnc(population[i, j, 0], 3, 1000, 0.35)
            population[i, j, 1] = rnc(population[i, j, 1], 3, 200, 0.35)
    return population


def main():
    # Inputs of the equation.
    # x, y, w, h
    blocks = [
        [20, 44, 200, 50],
        [234, 120, 70, 32],
        [200, 27, 70, 32],
    ]
    blocks = np.array(blocks, dtype=np.uint8)
    this_grid = Grid(rows=6)
    this_grid.visualize(blocks=blocks)

    # Number of the weights we are looking to optimize.

    """
    Genetic algorithm parameters:
        Mating pool size
        Population size
    """
    pop_size = 48
    num_parents_mating = 2
    num_offspring = pop_size - num_parents_mating
    population = initialize_population(blocks, pop_size)

    num_generations = 30
    best_agents = []
    for generation in range(num_generations):
        print("Generation : ", generation)
        # Measuring the fitness
        fitness = ga.cal_pop_fitness(population, this_grid, blocks)

        # Selecting the best parents in the population for mating.
        parents = ga.select_mating_pool(population, fitness,
                                        num_parents_mating)

        # Generating next generation using crossover.
        offspring_crossover = ga.crossover(parents, num_offspring)

        # Adding some variations to the offsrping using mutation.
        offspring_mutation = ga.mutation(offspring_crossover)

        # The best result in the current iteration.
        print("Best result : ", max(fitness))
        best_match_idx = numpy.where(fitness == numpy.max(fitness))
        print('Best idx: ', best_match_idx)
        print("Best solution : ", population[best_match_idx])
        print("Best solution fitness : ", fitness[best_match_idx])
        best_best_idx = best_match_idx[0] if len(best_match_idx[0]) > 1 else best_match_idx[0][0]

        very_best = population[best_best_idx].squeeze().copy()
        best_agents.append(very_best)
        if max(fitness[best_match_idx] == 1.0) or generation == num_generations - 1:
            print(f'Found earlier, generation: {generation}')
            im = this_grid.visualize(blocks=population[best_best_idx].squeeze(), show=True)
            imgs = [this_grid.visualize(agent) for agent in best_agents]
            im.save('test.gif', save_all=True, append_images=imgs)

            break


        # Creating the new population based on the parents and offspring.
        population[0:parents.shape[0], :] = parents
        population[parents.shape[0]:, :] = offspring_mutation

    # Getting the best solution after iterating finishing all generations.
    # At first, the fitness is calculated for each solution in the final generation.
    # fitness = ga.cal_pop_fitness(population, this_grid)
    # Then return the index of that solution corresponding to the best fitness.
    # best_match_idx = numpy.where(fitness == numpy.max(fitness))
    # print('Best idx: ', best_match_idx)
    # print("Best solution : ", population[best_match_idx])
    # print("Best solution fitness : ", fitness[best_match_idx])


if __name__ == '__main__':
    main()
