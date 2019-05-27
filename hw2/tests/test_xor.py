'''
零基础入门深度学习(3) - 神经网络和反向传播算法
    https://www.zybuluo.com/hanbingtao/note/476663
【机器学习】神经网络实现异或（XOR）
    https://www.cnblogs.com/Belter/p/6711160.html
'''

import numpy as np
import math

# data preprocess

# initialize variables
train_data = np.array([
    [0,0],
    [0,1],
    [1,0],
    [1,1]
])
train_label = np.array([
    0,1,1,0
])

def calcInitWeightRange(node_nums):
    return math.sqrt(node_nums[0]+node_nums[-1])
step = 1
node_nums = [2,5,1]
epochs = 1000
layer_num = len(node_nums)
wran = calcInitWeightRange(node_nums)
# wran=1

# What are good initial weights in a neural network?
#   https://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network?newreg=a23a99068f3049eeaf8b8eaf533974cf
weights = []
biases = []
for i in range(layer_num-1):
    # weights.append(np.zeros((node_nums[i],node_nums[i+1])))
    weights.append(np.random.rand(node_nums[i],node_nums[i+1]) * wran)
    biases.append(np.zeros(node_nums[i+1]))
    # biases.append(np.random.rand(node_nums[i+1]) * wran)

test_data = np.copy(train_data)
test_label = np.copy(train_label)


# train
def sigmoid(x):
    return 1 / (1+math.exp(-x))

def calcOutput(vin,target):
    global weights
    # initialize nodes values
    nodes = []
    deltas = []
    for i in range(layer_num):
        nodes.append(np.zeros(node_nums[i]))
        deltas.append(np.zeros(node_nums[i]))
    nodes[0] = vin

    # forward propagation
    for i in range(layer_num-1):
        for k in range(node_nums[i+1]):
            for j in range(node_nums[i]):
                nodes[i+1][k] += nodes[i][j] * weights[i][j,k]
            nodes[i+1][k] += biases[i][k]
            nodes[i+1][k] = sigmoid(nodes[i+1][k])

    ## back propagation
    '''
    输出层： delta = y*(1-y)*(t-y)
    隐藏层： delta = a*(1-a)*Sum(w*delta)
    w += step * delta * x(i,j)
    '''
    # calculate deltas
    for i in range(layer_num)[::-1]:
        for j in range(node_nums[i]):
            ytmp = nodes[i][j]
            if i == layer_num-1: # ouput layer
                deltas[i][j] = ytmp * (1-ytmp) * (target-ytmp)
            else: # hidden layers
                dtmp = 0
                for k in range(node_nums[i+1]):
                    dtmp += deltas[i+1][k] * weights[i][j,k]
                deltas[i][j] = ytmp * (1-ytmp) * dtmp

    # update weights
    for i in range(layer_num-1):
        for k in range(node_nums[i+1]):
            for j in range(node_nums[i]):
                weights[i][j,k] += step * deltas[i+1][k] * nodes[i][j]
            biases[i][k] += step * deltas[i+1][k]

    print(nodes[-1][0],end=" ")
# 每次从训练数据中取出一个样本的输入向量，使用感知器计算其输出，再根据上面的规则来调整权重。
# 每处理一个样本就调整一次权重。
# 经过多轮迭代后（即全部的训练数据被反复处理多轮），就可以训练出感知器的权重，使之实现目标函数。
for h in range(epochs):
    for i in range(train_data.shape[0]):
        vin = train_data[i]
        target = train_label[i]
        calcOutput(vin, target)
    print("")
print(weights)
print(biases)
