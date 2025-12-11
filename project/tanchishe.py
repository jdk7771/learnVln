import numpy as np
import random

class SnakeEnv():
    def __init__(self,grid_size = 20):
        self.map = np.zeros((20,20))
        self.bodysize = 3
        self.apple = (random.randint(1,19),random.randint(1,19))
        print(self.apple)


if __name__ == "__main__":
    a = SnakeEnv(10)
