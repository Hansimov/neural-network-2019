import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from random import randint
from shutil import copyfile

src = "data/"
dst = "dataX/"

def renameFiles(src,dst):
    for filename in os.listdir(src):
        name, ext = os.path.splitext(filename)
        print(name)
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
        copyfile(src+filename,dst+cls+filename)

imgs = []
def sharpImages(dst):
    global imgs
    for filename in os.listdir(dst):
        tmp = cv2.imread(dst+filename,0)
        # res = tmp
        # tmp = cv2.imread(src+filename)
        # res = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
        res = cv2.Sobel(tmp, cv2.CV_64F,1,1, ksize=3)
        # res = cv2.Laplacian(tmp,cv2.CV_64F,ksize=5)
        # res = cv2.Scharr(tmp,cv2.CV_64F,1,0)
        imgs.append([res,filename])

means = []
varis = []
names = []
def extractFeatures(imgs):
    global means, varis, names
    for img in imgs:
        pixs = img[0]
        # means.append(np.mean(pixs))
        # means.append(np.sum(pixs**2)/img.size)
        means.append(np.sum(np.absolute(pixs))/pixs.size)
        # varis.append(np.var(pixs))
        varis.append(np.std(pixs))
        names.append(img[1])
        # plt.scatter(means, varis)
    means = np.array(means)
    varis = np.array(varis)
    [means, varis] = list(map(lambda x: np.interp(x, (x.min(),x.max()),(0,10)), [means, varis]))
    plt.plot(means, varis, 'ro')
    plt.show()

def calcDistance(a,b):
    # a,b is n-d vector
    
    
    return dist

def clusterImages(k, imgs):
    clusters = [[] for _ in range(k)]

if __name__ == '__main__':
    # renameFiles(src,dst)
    sharpImages(dst)
    extractFeatures(imgs)
    # clusterImages(k, imgs)


