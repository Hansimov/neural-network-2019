import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
from shutil import copyfile
import pickle
import math
import time

data_original = "data/"
data_renamed = "data_renamed/"
data_processed = "data_processed/"
imgxs_labels = "imgxs_labels.pkl"

def renameFiles():
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
imgx_len = imgx_size[0] * imgx_size[1]
label_size = 4
# node_nums = [imgx_len, int(math.sqrt(imgx_len)), label_size]
node_nums = [imgx_len, 5, label_size]
layer_num = len(node_nums)
epochs = 50
step = 0.1
wran = step

def processFiles():
    # Resize images
    if not os.path.exists(data_processed): 
        os.mkdir(data_processed)
    for filename in os.listdir(data_renamed):
        img = cv2.imread(data_renamed+filename,0)
        # print(filename,img.shape, np.amax(img), np.amin(img))
        imgx = cv2.resize(img,imgx_size)
        body, ext = os.path.splitext(filename)
        cv2.imwrite(data_processed+body+"_x"+ext,imgx)

def dumpFiles():
    # Dump variables to files
    cnt = 0
    imgxs = []
    labels = []
    targets = []
    names = []
    for filename in os.listdir(data_processed):
        # print(filename)
        names.append(filename)
        imgx = cv2.imread(data_processed+filename,0)/255
        imgxs.append(imgx.flatten())
        label = ord(filename[0])-ord("A")
        labels.append(label)
        targets.append(onehot(label,label_size))
        # print(np.amax(imgx),np.amin(imgx))
        # print(imgx)

    with open(imgxs_labels, "wb") as wf:
        pickle.dump([names,imgxs,labels,targets], wf)

def nsigmoid(x):
    return 1 / (1+math.exp(-x))
sigmoid = np.vectorize(nsigmoid)

weights = []
biases = []

def trainNetwork():
    global weights, biases

    # Load imgxs and labels
    with open(imgxs_labels, "rb") as rf:
        names,imgxs,labels,targets = pickle.load(rf)
    # print(imgxs,labels,targets)

    # Initialize weights and biases
    for i in range(layer_num-1):
        weights.append(np.random.rand(node_nums[i],node_nums[i+1]) * wran)
        biases.append(np.zeros((1,node_nums[i+1])))

    # train through all samples
    for h in range(epochs):
        tic = time.time()
        total_count = len(imgxs)
        wrong_count = 0
        for i in range(len(imgxs)):
            # print("Epoch:{} Image:{:>4} Label:{} Target:{}".format(h,i,labels[i], targets[i]))
            is_wrong = trainSingleSample(names[i],imgxs[i], targets[i], labels[i])
            wrong_count += is_wrong
        toc = time.time()
        print("Epoch:{} == Time: {} s == Accuracy:{}/{}={}%".format(h,round(toc-tic,3),total_count-wrong_count,total_count,round(100*(1-wrong_count/total_count),2)))

def trainSingleSample(name, imgx, target, label):
    global weights, biases
    # Initialize nodes vales
    nodes = []
    deltas = []
    for i in range(layer_num):
        nodes.append(np.zeros((1,node_nums[i])))
        deltas.append(np.zeros((1,node_nums[i])))
    nodes[0] = imgx[np.newaxis]

    # forward propagation
    for i in range(layer_num-1):
        # for k in range(node_nums[i+1]):
            # for j in range(node_nums[i]):
            #     nodes[i+1][k] += nodes[i][j] * weights[i][j,k]
            # nodes[i+1][0,k] = np.dot(nodes[i],weights[i][:,k]) + biases[i][0,k]
            # nodes[i+1][0,k] = sigmoid(nodes[i+1][0,k])
        # nodes[i+1] = np.dot(nodes[i],weights[i][:,:]) + biases[i]

        nodes[i+1] = np.matmul(nodes[i], weights[i]) + biases[i]
        nodes[i+1] = sigmoid(nodes[i+1])

    target = target[np.newaxis]
    # calculate deltas
    for i in range(layer_num)[::-1]:
        if i == layer_num-1: # ouput layer
            deltas[i] = nodes[i]*(1-nodes[i])*(target-nodes[i])
        else: # hidden layer
            dtmp = np.matmul(weights[i],deltas[i+1].T)
            deltas[i] = nodes[i]*(1-nodes[i])*(dtmp.T)

        # for j in range(node_nums[i]):
        #     ytmp = nodes[i][0,j]
        #     # print(nodes[i].shape)
        #     if i == layer_num-1: # ouput layer
        #         deltas[i][0,j] = ytmp * (1-ytmp) * (target[j]-ytmp)
        #     else: # hidden layers
        #         # dtmp = 0
        #         # for k in range(node_nums[i+1]):
        #         #     dtmp += deltas[i+1][k] * weights[i][j,k]
        #         dtmp = np.dot(deltas[i+1], weights[i][j,:])
        #         deltas[i][0,j] = ytmp * (1-ytmp) * dtmp

    # update weights
    for i in range(layer_num-1):
        # for k in range(node_nums[i+1]):
        #     # for j in range(node_nums[i]):
        #     #     weights[i][j,k] += step * deltas[i+1][k] * nodes[i][j]
        #     weights[i][:,k] += step * np.dot(deltas[i+1][k], nodes[i])
        #     # biases[i][k] += step * deltas[i+1][k]
        weights[i] += step * np.matmul(nodes[i].T, deltas[i+1])
        biases[i] += step * deltas[i+1]

    # print(nodes[-1], np.argmax(nodes[-1]))
    if np.argmax(nodes[-1]) == label:
        is_wrong = 0
    else:
        is_wrong = 1
        # print(name)
    return is_wrong

if __name__ == '__main__':
    # renameFiles()
    # processFiles()
    # dumpFiles()
    trainNetwork()
