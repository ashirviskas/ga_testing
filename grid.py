from dataclasses import dataclass

import numpy as np


@dataclass
class Grid:
    cols: int = 12
    rows: int = 2
    col_gap: int = 24
    row_gap: int = 16
    row_size: int = 48
    width: int = 1200

    def __post_init__(self):
        self.calculate_grid()

    def calculate_grid(self):
        self.height = (self.rows * self.row_size) + (self.rows - 1) * self.row_gap
        self.np_grid = np.zeros((self.width, self.height))
        # upper left corner views
        self.np_grid[0::self.width // self.cols, 0::self.row_size + self.row_gap] = 1
        # a = self.np_grid.T
        # print(self.np_grid.shape)

    def find_nearest_point(self, indexes, point):
        x_dist = np.subtract(indexes[0], point[0])
        y_dist = np.subtract(indexes[1], point[1])
        tot_dist = np.power(np.power(x_dist, 2) + np.power(y_dist, 2), 1/2)
        idx = tot_dist.argmin()
        return idx, tot_dist[idx]

    def calculate_blocks_score(self, blocks):
        upper_left_corners = np.where(self.np_grid == 1)
        distances = []
        for block in blocks:
            idx, dist = self.find_nearest_point(upper_left_corners, block)
            distances.append(dist)

        return 1 / (sum(distances) + 1)



if __name__ == '__main__':
    griddy = Grid()
    print(griddy.calculate_blocks_score([[5, 5, 12, 12]]))
