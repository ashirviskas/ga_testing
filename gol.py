import numpy as np


class GameOfLife:
    def __init__(self, size):
        self.size = size
        self.grid = np.zeros(size, dtype=np.int)
        self.prepare_steps_thingies()

    def prepare_steps_thingies(self):
        directions = np.array([-1, 0, 1])
        rolls = np.stack([np.repeat(directions, 3), np.tile(directions, 3)])
        rolls = [[a, b] for a, b in rolls.T if a != 0 or b != 0]
        self.rolls = np.array(rolls)

    def get_similarity(self, other_gol):
        return (self.grid == other_gol.grid).sum()

    def random_populate(self, p=0.3):
        self.grid = np.random.choice(a=[True, False], size=self.size, p=[p, 1-p])

    def step(self):
        next_grid = np.zeros(self.size, dtype=np.int)
        neighbors = np.array([np.roll(self.grid, r, axis=(0, 1)) for r in self.rolls], dtype=np.int).sum(axis=0)
        # All with three neighbors
        next_grid[neighbors == 3] = 1
        # 2 neighbors + alive
        neighbors += self.grid
        next_grid[neighbors == 3] = 1
        self.grid = next_grid




def main():
    size = (10, 10)
    gol_a = GameOfLife(size=size)
    gol_a.random_populate()
    for i in range(10):
        gol_a.step()
        print(np.matrix(gol_a.grid))
    # gol_a.random_populate()
    gol_b = GameOfLife(size=size)
    # gol_b.random_populate()
    gol_a.step()

    print(gol_a.get_similarity(gol_b))

if __name__ == '__main__':
    main()
