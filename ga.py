import numpy as np
import random


# This project is extended and a library called PyGAD is released to build the genetic algorithm.
# PyGAD documentation: https://pygad.readthedocs.io
# Install PyGAD: pip install pygad
# PyGAD source code at GitHub: https://github.com/ahmedfgad/GeneticAlgorithmPython


def cal_pop_fitness(population, grid, original_blocks):
    fitness = np.zeros(population.shape[0])
    for i, blocks in enumerate(population):
        fitness[i] = grid.calculate_blocks_score(blocks, original_blocks)
    return fitness


def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = np.empty((num_parents, *pop[0].shape))
    for parent_num in range(num_parents):
        max_fitness_idx = np.where(fitness == np.max(fitness))[0]
        if len(max_fitness_idx) > 1:
            max_fitness_idx = max_fitness_idx[0]
        # max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num] = pop[max_fitness_idx]
        fitness[max_fitness_idx] = -99999999999
    return parents


def crossover(parents, num_offspring):
    offspring = np.empty((num_offspring, *parents[0].shape))

    for k in range(num_offspring):
        # Index of the first parent to mate.
        parent1_idx = k % parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k + 1) % parents.shape[0]

        mask = np.random.choice([0, 1], size=parents[0].shape, p=[0.5, 0.5])
        mask_inv = 1 - mask
        offspring[k, mask] = parents[parent1_idx, mask]
        offspring[k, mask_inv] = parents[parent2_idx, mask_inv]
        # The new offspring will have its second half of its genes taken from the second parent.

    return offspring


def mutation(pop):
    # Mutation changes a single gene in each offspring randomly.
    for idx, p in enumerate(pop):
        # The random value to be added to the gene.
        diff_mutation_mask = np.random.choice([0, 1], size=p.shape, p=[0.8, 0.2])
        abs_mutation_mask = np.random.choice([0, 1], size=p.shape, p=[0.9, 0.1])


        diff_mutations = np.random.randint(-10, 10, size=p.shape)
        abs_mutations = np.random.randint(0, 1000, size=p.shape)
        pop[idx] = p + (diff_mutation_mask * diff_mutations)
        pop[idx] = (abs_mutations * abs_mutation_mask) + (1 - abs_mutation_mask) * p
    return pop
