import numpy as np
from matplotlib import pyplot as plt
import ga
from gol import GameOfLife

class Evolution:
    def __init__(self, goal_gol, pop_size, num_parents_mating=2, num_generations=120, verbose=False, steps=3):
        self.goal_gol = goal_gol
        self.pop_size = pop_size
        self.num_parents_mating = num_parents_mating
        self.num_generations = num_generations
        self.verbose = verbose

        self.steps = steps

        self.num_offspring = pop_size - num_parents_mating

        self.population = None
        self.best_agents = []

        self.initialize_population()
        ga.seed_np_choices(goal_gol.size, p=0.1, amount_choices=1000)

    def initialize_population(self):
        self.population = [GameOfLife(self.goal_gol.size) for i in range(self.pop_size)]
        for p in self.population:
            p.random_populate()

    def get_pop_to_np(self):
        population_np = np.zeros((self.pop_size, *self.goal_gol.size), dtype=np.int)
        for i, p in enumerate(self.population):
            population_np[i] = p.grid
        return population_np

    def set_np_to_pop(self, population_np):
        for i, p in enumerate(population_np):
            self.population[i].grid = p

    def pop_do_steps(self):
        for p in self.population:
            p.do_steps(self.steps)

    def generation(self):
        # Measuring the fitness
        population_initials = self.get_pop_to_np()
        self.pop_do_steps()

        fitness = np.array([self.goal_gol.get_similarity(p) for p in self.population])

        # Selecting the best parents in the population for mating.
        parents = ga.select_mating_pool(population_initials, fitness, self.num_parents_mating)

        # Generating next generation using crossover.
        offspring_crossover = ga.crossover(parents, population_initials, self.num_offspring)

        # Adding some variations to the offsrping using mutation.
        offspring_mutation = ga.mutation(offspring_crossover)

        best_match_idx = np.argmax(fitness)
        best_fitness = fitness[best_match_idx]

        very_best = population_initials[best_match_idx].squeeze().copy()

        # Creating the new population based on the parents and offspring.
        population_initials[:parents.shape[0], :] = parents
        population_initials[parents.shape[0]:, :] = offspring_mutation
        self.best_agents.append(very_best)
        self.set_np_to_pop(population_initials)
        print(max(fitness))
        return max(fitness)

    def evolve(self, save_gif=False, non_increase_stop=5, min_generations=40):
        best_fitness = 0
        fitness_stagnant = 0
        results = []
        for i in range(self.num_generations):
            last_fitness = self.generation()
            results.append(last_fitness)
            if last_fitness > best_fitness:
                best_fitness = last_fitness
                fitness_stagnant = 0
            else:
                fitness_stagnant += 1

            # if i == self.num_generations - 1 or (fitness_stagnant == non_increase_stop and i > min_generations):
            #     print(f'Found earlier, generation: {i}')
            #     im = self.this_grid.visualize(blocks=self.best_agents[-1], show=False)
            #     if save_gif:
            #         imgs = [self.this_grid.visualize(agent) for agent in self.best_agents]
            #         im.save('test.gif', save_all=True, append_images=[im] * 3 + imgs)
            #     return results

def main():
    goal_gol = GameOfLife((10, 10))
    goal_gol.random_populate()
    goal_gol.do_steps(3)
    # goal_gol.grid[2, 1] = 1
    # goal_gol.grid[2, 2] = 1
    # goal_gol.grid[2, 3] = 1
    ev = Evolution(goal_gol=goal_gol, pop_size=120)
    ev.evolve()

if __name__ == '__main__':
    main()

