import numpy as np
import random
from enum import Enum
import time
import sys, select
import termios,tty


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
    up = (-1,0)
    left = (0,1)
    right = (0,-1)
    down = (1,0)    

turn_direc={
    0:Direction.up,
    1:Direction.left,
    2:Direction.down,
    3:Direction.right
}



class SnakeEnv():
    def __init__(self,grid_size = 20):
        self.grid_size = grid_size
        self.bodysize = 3

        self.reset()

    def reset(self):
        self.map = np.full((self.grid_size, self.grid_size), ' ', dtype='<U1')

        self.snake_head = (random.randint(3,self.grid_size-4),random.randint(3,self.grid_size-4))
        self.snake_body = [(self.snake_head[0],self.snake_head[1]+i) for i in range(self.bodysize)]

##运动方向
        self.direction = 3

        self.apple = (random.randint(1,self.grid_size-2),random.randint(1,self.grid_size-2))
        while( self.apple in self.snake_body):
            self.apple = (random.randint(1,self.grid_size-1),random.randint(1,self.grid_size-1))
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
    def update_observation(self):             
        self.observation = {
        "snake_head" : self.snake_head,
        "snake_body" : self.snake_body,
        "apple" : self.apple,
        "direction" : self.direction
    }   
        self.reward = 0
        


    def get_apple(self):
        if (self.snake_head == self.apple):
            self.apple = (random.randint(1,self.grid_size-2),random.randint(1,self.grid_size-2))
            self.all_reward += 1
            while( self.apple in self.snake_body):
                self.apple = (random.randint(1,self.grid_size-1),random.randint(1,self.grid_size-1))
            
            return 1
        else:
            return 0 

    def get_reward(self,action):
        self.reward = Ac2Re[action].value
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
        if self.snake_head[0]==0 or self.snake_head[0]>=self.grid_size-1  or  self.snake_head[1]==0 or self.snake_head[1]>=self.grid_size-1 or self.snake_head in [list(x) for x in self.snake_body[1:]]:
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
        self.get_reward(action)
        coll = self.checkcoll()
        self.update_observation()
        if coll== True:
            input("按下重新开始,选择done")
            self.done = True
            self.reset()
        
        return (self.observation, self.all_reward, self.done, self.info)
    
    def show_snak(self):
        self.map = np.full((self.grid_size, self.grid_size), ' ', dtype='<U1')  # 清空地图
        for (x,y) in self.snake_body:
            self.map[x][y] = 'o'
        self.map[self.snake_head[0]][self.snake_head[1]] = 'x'

        self.map[self.apple[0]][self.apple[1]] = 'p'

        for i in range(self.grid_size):
            self.map[0][i] = '——'
            self.map[self.grid_size-1][i] = '——'
            self.map[i][0] = '|'
            self.map[i][self.grid_size-1] = '|'
        for row in self.map:
            print(' '.join(row))

# def get_action(timeout=0.3):
#     """timeout 秒内读取一次输入，没输入返回0"""
#     i, o, e = select.select([sys.stdin], [], [], timeout)
#     if i:
#         key = sys.stdin.read(1)  # 读一个字符
#         if key == 'w':
#             return 1  # 上
#         elif key == 's':
#             return 0  # 下
#         elif key == 'a':
#             return 2  # 左
#         elif key == 'd':
#             return 3  # 右
#     return 0

def get_action_unix(timeout=0.3):
    import select
    i, o, e = select.select([sys.stdin], [], [], timeout)
    if i:
        key = sys.stdin.read(1)
        # 映射键盘 w/a/d 到你的 Action 定义
        # Action: 0:直行, 1:加速, 2:右转, 3:左转
        print(key)   
        if key == 'w': return 1 # 加速
        if key == 'd': return 2 # 右转
        if key == 'a': return 3 # 左转
        if key == 's': return 0 # 直行

    return 0


if __name__ == "__main__":
    a = SnakeEnv(20)  
    
    # 1. 保存旧的终端设置 (为了程序退出后恢复终端)
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    
    try:
        # 2. 将终端设置为 cbreak 模式 (按键即响应，不需要回车)
        tty.setcbreak(fd)
        
        while not a.done:
            # 这里简化了你的逻辑：
            # 直接利用 select 的 timeout 作为游戏的帧间隔
            # 如果 timeout 时间内有输入，立刻执行；如果没有，返回0继续走
            
            # 这里的 0.3 就是游戏的刷新速度 (难度)
            action = get_action_unix(timeout=0.3) 
            # input("")
            # 执行一步
            a.step(action)
            print(round(a.all_reward,1))
            a.show_snak()

    except KeyboardInterrupt:
        print("\n退出游戏")
        
    finally:
        # 3. 无论程序是否报错，最后都要恢复终端设置
        # 否则你退出程序后，终端会乱掉（看不到输入的内容）
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)