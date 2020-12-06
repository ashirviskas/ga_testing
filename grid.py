from dataclasses import dataclass
from PIL import Image, ImageDraw
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
        self.height = (self.rows * self.row_size) + (self.rows - 1) * self.row_gap + 1
        self.np_grid = np.zeros((self.width, self.height))
        self.np_grid[:, :] = -1
        self.col_size = ((self.width - (self.cols - 1) * self.col_gap) // self.cols)

        # upper left corner views
        self.np_grid[0::self.col_size + self.col_gap, 0::self.row_size + self.row_gap] = 0
        self.upper_left_corners = np.where(self.np_grid == 0)

        # upper right corner views
        self.np_grid[self.col_size::self.col_size + self.col_gap, 0::self.row_size + self.row_gap] = 1
        self.upper_right_corners = np.where(self.np_grid == 1)

        # bottom left corner views
        self.np_grid[::self.col_size + self.col_gap, self.row_size::self.row_size + self.row_gap] = 2
        self.bottom_left_corners = np.where(self.np_grid == 2)

        # bottom right corner views
        self.np_grid[self.col_size::self.col_size + self.col_gap, self.row_size::self.row_size + self.row_gap] = 3
        self.bottom_right_corners = np.where(self.np_grid == 3)

        # img = self.np_grid.T.copy()
        # img = ((img + 1) / img.max()) * 255
        # img = Image.fromarray(img)
        # img.show()
        # a = self.np_grid.T
        # print(self.np_grid.shape)

    def find_nearest_point(self, point):
        x_left = point[0] % (self.col_size + self.col_gap)
        x_right = (point[0] + self.col_size) % (self.col_size + self.col_gap)
        x_dist = min(x_left, x_right)
        y_top = point[1] % (self.row_size + self.row_gap)
        y_bottom = (point[1] + self.row_size) % (self.row_size + self.row_gap)
        y_dist = min(y_top, y_bottom)
        tot_dist = np.power(np.power(x_dist, 2) + np.power(y_dist, 2), 1 / 2)
        return tot_dist

    @staticmethod
    def dist_to_score(dist):
        if dist < 2:
            return 30
        else:
            return -dist

    def calculate_blocks_score(self, blocks, original_blocks):
        scores = []

        total_diff = abs(blocks - original_blocks).sum()
        scores.append(-total_diff/10)
        # upper_left_corners = np.where(self.np_grid == 0)
        for block in blocks:
            dist = self.find_nearest_point(block)
            score = self.dist_to_score(dist)
            scores.append(score)
        total_score = sum(scores)
        return total_score

    def visualize(self, blocks=None, show=False):
        img = Image.new('RGBA', self.np_grid.shape)
        draw = ImageDraw.Draw(img)

        for x, y in zip(*self.upper_left_corners):
            draw.rectangle((x, y, x + self.col_size, y + self.row_size), fill=(180, 180, 160, 160))
        if blocks is not None:
            for block in blocks:
                draw = ImageDraw.Draw(img)
                draw.rectangle((block[0], block[1], block[0] + block[2], block[1] + block[3]), fill=(120, 255, 255, 160))
        if show:
            img.show()
        return img


if __name__ == '__main__':
    griddy = Grid()
    # print(griddy.calculate_blocks_score([[5, 5, 12, 12]]))
    griddy.visualize(blocks=[[20, 40, 200, 30]])
