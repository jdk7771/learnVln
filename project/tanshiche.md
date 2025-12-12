好，Challenge Accepted（挑战接受）。

请关闭之前的所有代码窗口。新建一个空白文件，命名为 snake_hard_mode.py。

既然是 Hard 模式，我就不再给你代码填空了。你是架构师，你是开发人员。

我只作为**“甲方”（或你的导师），给你发一份《技术需求文档 (PRD)》**。你需要根据这份文档，写出符合工业标准的 Simulation Class。

📋 甲方需求文档 (Project: VLN-Snake-Sim)
项目目标：开发一个符合 OpenAI Gym 标准的 2D 导航环境。 交付物：一个包含 SnakeEnv 类的 Python 文件。

1. 类结构要求 (Class Interface)
你必须严格实现以下 API，函数名、参数、返回值类型必须完全一致，否则视为验收不通过（因为自动化测试脚本会跑不通）。

__init__(self, grid_size=10)

初始化地图大小。

初始化蛇头、蛇身、食物。

定义 self.action_space (int)。

定义 self.observation_space (tuple)。

reset(self)

功能：重置环境状态。

返回：observation (类型自定义，建议是字典或numpy数组)。

step(self, action)

参数：action (int)，取值 0-3。

功能：计算下一帧的物理状态。

返回：一个包含 4 个元素的 Tuple (observation, reward, done, info)。

observation: 当前状态（同 reset 的返回值）。

reward (float): 吃到食物 +1，撞墙/撞自己 -1，普通走步 0（或者 -0.01 鼓励快走）。

done (bool): 游戏是否结束。

info (dict): 调试信息，可以为空 {}。
S
render(self)

功能：打印当前网格的字符画。

约束：严禁在 step 函数里写 print，所有的 print 必须写在这里。

2. 核心物理逻辑要求 (Physics Engine)
这是最考验你逻辑思维的地方，很多新手写不出来：

蛇身跟随：当蛇头移动到 (x, y) 时，原来的蛇头位置应该变成身体的第一节，原来的身体最后一节应该移出（除非吃到了食物）。

提示：使用 Python 的 list.insert(0, head) 和 list.pop() 是最高效的方法。

坐标系定义：

建议使用 矩阵坐标系：x 代表行 (row)，y 代表列 (col)。

左上角是 (0, 0)，右下角是 (grid_size-1, grid_size-1)。

碰撞检测：

Wall Collision: x < 0 或 x >= size ...

Self Collision: 新的蛇头坐标是否已经存在于 self.body 列表中？

🧪 验收测试脚本 (The Examiner)
等你写完你的 Class 后，不要修改下面的代码，直接把这段代码粘贴到你的文件最下方运行。

如果你的类写得完美，这段代码会流畅运行，你会可以用键盘玩游戏。如果报错，说明你的接口没对齐，或者逻辑有 Bug。