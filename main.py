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


class Evolution:
    def __init__(self, blocks, this_grid, pop_size, num_parents_mating=2, num_generations=120):
        self.blocks = blocks
        self.this_grid = this_grid
        self.pop_size = pop_size
        self.num_parents_mating = num_parents_mating
        self.num_generations = num_generations

        self.num_offspring = pop_size - num_parents_mating

        self.population = None
        self.best_agents = []

        self.initialize_population()

    def initialize_population(self,):
        self.population = np.tile(self.blocks, (self.pop_size, 1, 1))
        for i in range(1, len(self.population)):
            diff_mutation_mask = np.random.choice([0, 1], size=self.blocks.shape, p=[0.5, 0.5])
            diff_mutations = np.random.randint(-10, 10, size=self.population[i].shape)
            self.population[i, diff_mutation_mask] = np.abs(self.population[i] + diff_mutations)[diff_mutation_mask]

    def generation(self):
        # Measuring the fitness
        fitness = ga.cal_pop_fitness(self.population, self.this_grid, self.blocks)

        # Selecting the best parents in the population for mating.
        parents = ga.select_mating_pool(self.population, fitness, self.num_parents_mating)

        # Generating next generation using crossover.
        offspring_crossover = ga.crossover(parents, self.population, self.num_offspring)

        # Adding some variations to the offsrping using mutation.
        offspring_mutation = ga.mutation(offspring_crossover)

        # The best result in the current iteration.
        print("Best result : ", max(fitness))
        print("Average result : ", sum(fitness) / len(fitness))
        best_match_idx = np.argmax(fitness)
        best_fitness = fitness[best_match_idx]
        print('Best idx: ', best_match_idx)
        print("Best solution : ", self.population[best_match_idx])
        print("Best solution fitness : ", best_fitness)

        very_best = self.population[best_match_idx].squeeze().copy()

        # Creating the new population based on the parents and offspring.
        self.population[:parents.shape[0], :] = parents
        self.population[parents.shape[0]:, :] = offspring_mutation
        self.best_agents.append(very_best)

    def evolve(self):
        for i in range(self.num_generations):
            self.generation()

            if i == self.num_generations - 1:
                print(f'Found earlier, generation: {i}')
                im = self.this_grid.visualize(blocks=self.best_agents[-1], show=True)
                imgs = [self.this_grid.visualize(agent) for agent in self.best_agents]
                im.save('test.gif', save_all=True, append_images=[im] * 3 + imgs)


def main():
    # Inputs of the equation.
    # x, y, w, h

    blocks = [
        [20, 44, 200, 50],
        [234, 120, 70, 32],
        [200, 27, 70, 32],
    ]
    blocks = np.array(blocks, dtype=np.int)
    this_grid = Grid(rows=6)
    initial_img = this_grid.visualize(blocks=blocks, show=False)
    initial_img.save('initial_img.png')
    # Number of the weights we are looking to optimize.

    """
    Genetic algorithm parameters:
        Mating pool size
        Population size
    """
    pop_size = 12
    num_parents_mating = 2


    num_generations = 120

    ev = Evolution(blocks, this_grid, pop_size, num_parents_mating, num_generations)
    ev.evolve()

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
