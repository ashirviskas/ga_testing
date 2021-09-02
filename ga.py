import numpy as np
import random


# This project is extended and a library called PyGAD is released to build the genetic algorithm.
# PyGAD documentation: https://pygad.readthedocs.io
# Install PyGAD: pip install pygad
# PyGAD source code at GitHub: https://github.com/ahmedfgad/GeneticAlgorithmPython

NP_CHOICES = None


def seed_np_choices(shape, p, amount_choices=1000):
    global NP_CHOICES
    shape = (amount_choices, *shape)
    NP_CHOICES = np.random.choice([0, 1], size=shape, p=[1-p, p])



def cal_pop_fitness(population, grid, original_blocks):
    fitness = np.zeros(population.shape[0])
    for i, blocks in enumerate(population):
        fitness[i] = grid.calculate_blocks_score(blocks, original_blocks)
    return fitness


def select_mating_pool(pop, fitness, num_parents):
    fitness = fitness.copy()
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = np.empty((num_parents, *pop[0].shape), dtype=np.int)
    min_fitness = np.min(fitness)
    for parent_num in range(num_parents):
        max_fitness_idx = np.argmax(fitness)
        parents[parent_num] = pop[max_fitness_idx]
        fitness[max_fitness_idx] = min_fitness
    return parents


def choose_random(sample):
    return sample[np.random.choice(len(sample))]


def crossover(leaders, population, num_offspring):
    offspring = np.empty((num_offspring, *leaders[0].shape), dtype=np.int)
    leader_chance = 0.8
    for k in range(num_offspring):
        parents = []
        for i in range(2):
            parent = choose_random(leaders) if random.random() < leader_chance else choose_random(population)
            parents.append(parent)

        mask = np.random.choice([0, 1], size=leaders[0].shape, p=[0.5, 0.5])
        mask_inv = 1 - mask
        offspring[k] = parents[0] * mask + parents[1] * mask_inv
        # The new offspring will have its second half of its genes taken from the second parent.

    return offspring


def get_mask(shape):
    global NP_CHOICES
    selections = np.random.randint(0, len(NP_CHOICES))
    mask = NP_CHOICES[selections]
    return mask


def mutation(pop):
    # Mutation changes a single gene in each offspring randomly.
    for idx, p in enumerate(pop):
        # The random value to be added to the gene.
        mutation_mask = get_mask(p.shape)
        pop[idx] = p != mutation_mask
        # pop[idx] = (abs_mutations * abs_mutation_mask) + (1 - abs_mutation_mask) * p
    return pop
