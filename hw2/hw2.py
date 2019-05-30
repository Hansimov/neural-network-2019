'''
No.118039910141
于泽汉
2019.05
'''
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
import shutil
import pickle
import math
import time

data_original = "data/"
data_renamed = "data_renamed/"
data_train = "data_train/"
data_test = "data_test/"
dumped_train_data = "train_data.pkl"
dumped_test_data = "test_data.pkl"
dumped_weights = "weights.pkl"

def renameFiles():
    if os.path.exists(data_renamed): 
        shutil.rmtree(data_renamed)
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
        shutil.copyfile(data_original+filename,data_renamed+cls+filename)


imgx_size = (10,10)
imgx_len = imgx_size[0] * imgx_size[1]
label_size = 4
# node_nums = [imgx_len, int(math.sqrt(imgx_len)), label_size]
node_nums = [imgx_len, 5, label_size]
layer_num = len(node_nums)
epochs = 100
step = 0.1
wran = step
noise_max = 0.2 * 255

# aleju/imgaug: Image augmentation for machine learning experiments.
#   https://github.com/aleju/imgaug
#   http://imgaug.readthedocs.io

def generateTrainData():
    if os.path.exists(data_train): 
        shutil.rmtree(data_train)
    os.mkdir(data_train)

    names,imgxs,labels,targets = [],[],[],[]
    for filename in os.listdir(data_renamed):
        print(">>> Generating train data for {}".format(filename))
        if filename[0] == "D":
            aug_num = 500
        else:
            aug_num = 4
        for i in range(aug_num):
            # name
            body, ext = os.path.splitext(filename)
            name = data_train+body+"_x{:0>2}".format(i)+ext
            names.append(name)
            # imgx
            imgx = cv2.imread(data_renamed+filename,0)
            imgx = cv2.resize(imgx,imgx_size)
            noise = noise_max * np.random.rand(imgx_size[0],imgx_size[1])
            imgx = imgx + noise
            cv2.imwrite(name,imgx)
            imgx = imgx/255
            imgx = imgx.flatten()
            imgxs.append(imgx)
            # label
            label = ord(filename[0])-ord("A")
            labels.append(label)
            # target
            target = onehot(label,label_size)
            targets.append(target)

    # dump train data
    with open(dumped_train_data, "wb") as wf:
        pickle.dump([names,imgxs,labels,targets], wf)

def generateTestData():
    if os.path.exists(data_test): 
        shutil.rmtree(data_test)
    os.mkdir(data_test)

    names,imgxs,labels,targets = [],[],[],[]
    for filename in os.listdir(data_renamed):
        # name
        body, ext = os.path.splitext(filename)
        name = data_test+body+"_y"+ext
        names.append(name)
        # imgx
        imgx = cv2.imread(data_renamed+filename,0)
        imgx = cv2.resize(imgx,imgx_size)
        cv2.imwrite(name,imgx)
        imgx = imgx/255
        imgx = imgx.flatten()
        imgxs.append(imgx)
        # label
        label = ord(filename[0])-ord("A")
        labels.append(label)
        # target
        target = onehot(label,label_size)
        targets.append(target[np.newaxis])

    # dump test data
    with open(dumped_test_data, "wb") as wf:
        pickle.dump([names,imgxs,labels,targets], wf)


weights = []
biases = []
def trainNetwork():
    global weights, biases

    # load imgxs and labels
    with open(dumped_train_data, "rb") as rf:
        names,imgxs,labels,targets = pickle.load(rf)
    # print(imgxs,labels,targets)

    # initialize weights and biases
    for i in range(layer_num-1):
        weights.append(np.random.rand(node_nums[i],node_nums[i+1]) * wran)
        biases.append(np.zeros((1,node_nums[i+1])))

    accuracies = []
    # train through all samples
    for h in range(epochs):
        tic = time.time()
        total_count = len(imgxs)
        wrong_count = 0
        for i in range(len(imgxs)):
            is_wrong = trainSingleSample(names[i],imgxs[i], labels[i], targets[i])
            wrong_count += is_wrong
        toc = time.time()
        accuracy = round(100*(1-wrong_count/total_count),2)
        accuracies.append(accuracy)
        print("Epoch:{} == Time: {} s == Accuracy:{}/{}={}%".format(h,round(toc-tic,3),total_count-wrong_count,total_count,accuracy))
    plt.plot(list(range(len(accuracies))),accuracies)
    plt.ylabel("Accuracy of training data")
    plt.xlabel("Epoch")
    plt.show()

    # dump weights and biases
    with open(dumped_weights, "wb") as wf:
        pickle.dump([node_nums,layer_num,weights,biases], wf)

def nsigmoid(x):
    return 1 / (1+math.exp(-x))
sigmoid = np.vectorize(nsigmoid)

def trainSingleSample(name, imgx, label, target):
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
        nodes[i+1] = sigmoid(np.matmul(nodes[i], weights[i]) + biases[i])

    # calculate deltas
    for i in range(layer_num)[::-1]:
        if i == layer_num-1: # ouput layer
            deltas[i] = nodes[i]*(1-nodes[i])*(target-nodes[i])
        else: # hidden layer
            dtmp = np.matmul(weights[i],deltas[i+1].T)
            deltas[i] = nodes[i]*(1-nodes[i])*(dtmp.T)

    # update weights
    for i in range(layer_num-1):
        weights[i] += step * np.matmul(nodes[i].T, deltas[i+1])
        biases[i] += step * deltas[i+1]

    # check output label
    if np.argmax(nodes[-1]) == label:
        is_wrong = 0
    else:
        is_wrong = 1
    return is_wrong

def testNetwork():
    # load test data
    with open(dumped_test_data, "rb") as rf:
        names,imgxs,labels,targets = pickle.load(rf)
    # load trained weights
    with open(dumped_weights, "rb") as rf:
        node_nums,layer_num,weights,biases = pickle.load(rf)

    # test data
    total_count = len(imgxs)
    wrong_count = 0
    tic = time.time()
    for i in range(len(imgxs)):
        name,imgx,label,target = names[i],imgxs[i], labels[i], targets[i]
        # Initialize nodes vales
        nodes = []
        for i in range(layer_num):
            nodes.append(np.zeros((1,node_nums[i])))
        nodes[0] = imgx[np.newaxis]
        # forward propagation
        for i in range(layer_num-1):
            nodes[i+1] = np.matmul(nodes[i], weights[i]) + biases[i]
            nodes[i+1] = sigmoid(nodes[i+1])
        # check output label
        if np.argmax(nodes[-1]) == label:
            is_wrong = 0
        else:
            is_wrong = 1
            wrong_count += 1
            # print(name)
    toc = time.time()
    print("Time: {} s -- Accuracy:{}/{}={}%".format(round(toc-tic,3),total_count-wrong_count,total_count,round(100*(1-wrong_count/total_count),2)))

if __name__ == '__main__':
    # renameFiles()
    # generateTrainData()
    # generateTestData()
    trainNetwork()
    testNetwork()