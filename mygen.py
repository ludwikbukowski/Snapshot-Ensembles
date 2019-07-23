
import numpy as np

class BatchGenerator(object):
    def __init__(self, image_gen, len):
      self.image_gen = image_gen
      self.len = len

    def __iter__(self):
      return self


    def __next__(self):
      return self.next()

    def myfunc(self, a, len):
        return np.array([a for _ in range(len)])

    def next(self):
        (x_batch, y_batch) = self.image_gen.next()
        return [x_batch for _ in range(self.len)], y_batch



