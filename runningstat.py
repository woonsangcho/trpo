
''' modified from https://github.com/liyanage/python-modules/blob/master/running_stats.py,
 detailed in http://www.johndcook.com/standard_deviation.html'''

import numpy as np
class RunningStats:
    def __init__(self, dim):
        self.n = 0

        self.dim = dim
        self.old_m = np.zeros(dim)
        self.new_m = np.zeros(dim)
        self.old_s = np.zeros(dim)
        self.new_s = np.zeros(dim)

    def clear(self):
        self.n = 0

    def push(self, x):
        self.n += 1

        if self.n == 1:
            self.old_m = self.new_m = x
            self.old_s = np.zeros(self.dim)
        else:
            self.new_m = self.old_m + (x - self.old_m) / self.n
            self.new_s = self.old_s + (x - self.old_m) * (x - self.new_m)

            self.old_m = self.new_m
            self.old_s = self.new_s

    def mean(self):
        return self.new_m if self.n else 0.0

    def variance(self):
        return self.new_s / (self.n - 1) if self.n > 1 else 1.0

    def standard_deviation(self):
        return np.sqrt(self.variance()) + 1e-06

    def multiple_push(self, elements):
        for element in elements:
            self.push(element)
