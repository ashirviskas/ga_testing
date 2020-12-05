import numpy as np


class Block:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def to_np(self):
        return np.array([self.x, self.y, self.w, self.h], dtype=np.uint8)

    def from_np(self, np_array):
        self.x = np_array[0]
        self.y = np_array[1]
        self.w = np_array[2]
        self.h = np_array[3]
