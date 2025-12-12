import numpy as np
import random
from enum import Enum
import time
import sys, select


class Action(Enum):
    forward = 1
    turn_left = 3
    turn_right = 2
    none = 0
class Reward_action(Enum):
    forward = 0.1
    turn_left = 0
    turn_right = 0
    none = 0
Ac2Re= {
    0:Reward_action.none,
    1:Reward_action.forward,
    2:Reward_action.turn_right,
    3:Reward_action.turn_left
}



class Direction(Enum):
    up = (0,-1)
    left = (1,0)
    right = (-1,0)
    down = (0,1)    

turn_direc={
    0:Direction.up,
    1:Direction.left,
    2:Direction.down,
    3:Direction.right
}




class SnakeEnv():
    def __init__(self,grid_size = 20):
        self.grad_size = grid_size

        self.map = np.full((self.grad_size, self.grad_size), ' ', dtype='<U1')
        self.bodysize = 3
        self.snake_head = (random.randint(3,self.grad_size-4),random.randint(3,self.grad_size-4))
        self.snake_body = [(self.snake_head[0],self.snake_head[1]+i) for i in range(self.bodysize)]

##运动方向
        self.direction = 3

        self.apple = (random.randint(1,self.grad_size-2),random.randint(1,self.grad_size-2))
        while( self.apple in self.snake_body):
            self.apple = (random.randint(1,self.grad_size-1),random.randint(1,self.grad_size-1))
        self.action_space = 4
##这地方可以更新吗
        self.observation = {
            "snake_head" : self.snake_head,
            "snake_body" : self.snake_body,
            "apple" : self.apple,
            "direction" : self.direction
        }
        self.reward = 0
        self.all_reward = 0
        self.done = False

        self.info = ""

##恢复初始状态
    def reset(self):
        self.__init__(self.grad_size)

    def get_apple(self):
        if (self.snake_head == self.apple):
            self.apple = (random.randint(1,self.grad_size-2),random.randint(1,self.grad_size-2))
            while( self.apple in self.snake_body):
                self.apple = (random.randint(1,self.grad_size-1),random.randint(1,self.grad_size-1))
            
            return 1
        else:
            return 0 

    def get_reward(self,action):
        self.reward = Reward_action(Ac2Re(action))
        self.all_reward +=self.reward


    def do_usual(self):
        dx,dy = turn_direc[self.direction].value
        self.snake_head = (self.snake_head[0] + dx, self.snake_head[1] + dy)
        eat = self.get_apple()
        self.snake_body.insert(0,self.snake_head)
        if eat:
            pass
        else:
            self.snake_body.pop()


    def do_forward(self):
        self.do_usual()
        self.do_usual()


    def do_turn(self,dir):
        if dir==2:
            self.direction = (self.direction+1)%4
        elif dir ==3:
            self.direction = (self.direction-1)%4
        self.do_usual()


    def checkcoll(self):
        if self.snake_head[0]==0 or self.snake_head[0]==self.grad_size  or  self.snake_head[1]==0 or self.snake_head[1]==self.grad_size:
            return True
        else:
            return False


#输入一个action 空闲 向前 右转 左转 0\1\2\3
    def step(self,action=0):

        if action == 0:
            self.do_usual()
        elif action ==1:
            self.do_forward()
        else:
            self.do_turn(action)

        coll = self.checkcoll()
        if coll== True:
            input("按下重新开始,选择done")
            self.done = True
            self.reset()
        
        return (self.observation, self.reward, self.done, self.info)
    
    def show_snak(self):
        self.map = np.full((self.grad_size, self.grad_size), ' ', dtype='<U1')  # 清空地图
        for (x,y) in self.snake_body:
            self.map[x][y] = 'o'
        self.map[self.snake_head[0]][self.snake_head[1]] = 'x'

        self.map[self.apple[0]][self.apple[1]] = 'p'

        for i in range(self.grad_size):
            self.map[0][i] = '——'
            self.map[self.grad_size-1][i] = '——'
            self.map[i][0] = '|'
            self.map[i][self.grad_size-1] = '|'
        for row in self.map:
            print(' '.join(row))

def get_action(timeout=0.3):
    """timeout 秒内读取一次输入，没输入返回0"""
    i, o, e = select.select([sys.stdin], [], [], timeout)
    if i:
        key = sys.stdin.read(1)  # 读一个字符
        if key == 'w':
            return 1  # 上
        elif key == 's':
            return 0  # 下
        elif key == 'a':
            return 2  # 左
        elif key == 'd':
            return 3  # 右
    return 0


if __name__ == "__main__":
    a = SnakeEnv(20)  
    timeout = 0.5
    i=0
    while(True):
        action = 0
        time_start = time.time()
        while((time.time()-time_start) < timeout):
            action = get_action()
            if action!=0:
                break
        a.step(action)

        a.show_snak()
