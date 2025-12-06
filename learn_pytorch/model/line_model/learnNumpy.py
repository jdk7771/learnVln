import numpy as np


data = np.array([1,2,3])

data2 = np.array([[1,2,3,4],[2,3,4,5]])

data3 = np.zeros(shape= (3,4))

data4 = np.ones(shape= (10,4))

data5 = np.empty(shape=(3,4))

data6 = np.arange(10,20,2)

data7 = np.linspace(1,10,20)#1-10 平均分为20份

data8 = np.random.rand(3,4)

data9 = np.random.randint(1,5,size=(6,7))

dataVstack = np.vstack((data4, data3))
# dataHstack = np.hstack((data4, data3))
print(dataVstack)
# print(dataHstack)
