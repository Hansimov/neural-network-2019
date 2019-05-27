import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
from shutil import copyfile
import pickle
import math

data_original = "data/"
data_renamed = "data_renamed/"
data_processed = "data_processed/"
imgxs_labels = "imgxs_labels.pkl"

def renameFiles():
    # print(">>> Renaming files ...")
    if not os.path.exists(data_renamed): 
        os.mkdir(data_renamed)
    for filename in os.listdir(data_original):
        name, ext = os.path.splitext(filename)
        # print(name)
        if ext == ".bmp":
            cls = "A"
        elif ext == ".png":
            cls = "B"
        elif ext == ".jpg":
            if name == "timg":
                cls = "D"
            else:
                cls = "C"
        else:
            cls = "X"
        copyfile(data_original+filename,data_renamed+cls+filename)

def onehot(idx,len):
    tmp = np.zeros(len)
    tmp[idx] = 1
    return tmp

imgx_size = (10,10)
label_size = 4
def processFiles():
    # print(">>> Processing files ...")
    if not os.path.exists(data_processed): 
        os.mkdir(data_processed)
    cnt = 0
    # for filename in os.listdir(data_renamed):
    #     if cnt == 1:
    #         break
    #     cnt += 1
        # img = cv2.imread(data_renamed+filename,0)
        # number range: [0,255]
        # print(filename,img.shape, np.amax(img), np.amin(img))
        # imgx = cv2.resize(img,imgx_size)
        # body, ext = os.path.splitext(filename)
        # cv2.imwrite(data_processed+body+"_x"+ext,imgx)
    cnt = 0
    imgxs = []
    labels = []
    targets = []
    for filename in os.listdir(data_processed):
        # if cnt == 1:
        #     break
        # cnt += 1 
        # print(filename)
        imgx = cv2.imread(data_processed+filename,0)
        imgxs.append(imgx.flatten())
        labels.append(filename[0])
        targets.append(onehot(ord(filename[0])-ord("A"),label_size))
        # print(np.amax(imgx),np.amin(imgx))
        # print(imgx)
        
    with open(imgxs_labels, "wb") as wf:
        pickle.dump([imgxs,labels,targets], wf)
        # with open('imgdata.pkl', 'rb') as f:
        #     imgx = pickle.load(f)
            # print(imgx)

def calcInitWeightRange(node_nums):
    return math.sqrt(node_nums[0]+node_nums[1])
def sigmoid(x):
    return 1 / (1+math.exp(-x))

weights = []
biases = []
step = 1
node_nums = [imgx_size[0]*imgx_size[1], 10, label_size]
epochs = 100
layer_num = len(node_nums)
wran = calcInitWeightRange(node_nums)

def trainNetwork():
    global weights, biases

    # Load imgxs and labels
    with open(imgxs_labels, "rb") as rf:
        imgxs,labels,targets = pickle.load(rf)
    # print(imgxs,labels)

    # Initialize weights and biases
    for i in range(layer_num-1):
        weights.append(np.random.rand(node_nums[i],node_nums[i+1]) * wran)
        biases.append(np.zeros(node_nums[i+1]))

    # train through all samples
    for h in range(epochs):
        for i in range(len(imgxs)):
            print("Epoch:{} Image:{:>4} Label:{} Target:{}".format(h,i,labels[i], targets[i]))
            trainSingleSample(imgxs[i], targets[i])

def trainSingleSample(vin, target):
    global weights, biases
    # Initialize nodes vales
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
            nodes[i+1][k] = sigmoid(nodes[i+1][k]/255)

    print(nodes[-1])

    # calculate deltas
    for i in range(layer_num)[::-1]:
        for j in range(node_nums[i]):
            ytmp = nodes[i][j]
            if i == layer_num-1: # ouput layer
                deltas[i][j] = ytmp * (1-ytmp) * (target[j]-ytmp)
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

    print(nodes[-1])

if __name__ == '__main__':
    # renameFiles()
    # processFiles()
    trainNetwork()
