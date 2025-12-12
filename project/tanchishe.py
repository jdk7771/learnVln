import numpy as np
import random
from enum import Enum

class action(Enum):
    forward = 0
    turn_left = 1
    turen_right = 2

class SnakeEnv():
    def __init__(self,grid_size = 20):
        self.grad_size = grid_size

        self.map = np.zeros((self.grad_size ,self.grad_size ))
        self.bodysize = 3
        self.apple = (random.randint(1,19),random.randint(1,19))
        

    def reset(self):
        self.__init__(self.grad_size)


    def action(self,acti):
        if acti == 1:
            
        return (observation, reward, done, info)

if __name__ == "__main__":
    a = SnakeEnv(10)
