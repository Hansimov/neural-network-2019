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
        pix = img[0]
        # means.append(np.mean(pix))
        # means.append(np.sum(pix**2)/img.size)
        means.append(np.sum(np.absolute(pix))/pix.size)
        # varis.append(np.var(pix))
        varis.append(np.std(pix))
        names.append(img[1])
        # plt.scatter(means, varis)
    means = np.array(means)
    varis = np.array(varis)
    [means, varis] = list(map(lambda x: np.interp(x, (x.min(),x.max()),(0,10)), [means, varis]))
    plt.plot(means, varis, 'ro')
    # plt.show()

def calcDistance(a,b):
    # a,b is n-d vector
    dist = 0
    return dist

def clusterImages(k, imgs):
    n = len(means)
    mv = []
    for idx in range(0,n):
        mv.append(np.array([means[idx],varis[idx]]))

    visited = []
    clusters = [[] for _ in range(k)]
    clusters[0].append(randint(0,n-1))
    visited.append(clusters[0][0])

    # Initialize centers
    tmpCenter = np.array(mv[0])
    for i in range(1,k):
        maxDist = 0
        cenIdx = -1
        for j in range(0,n):
            if j in visited:
                pass
            else:
                tmpDist = np.linalg.norm(mv[j]-tmpCenter)
                if tmpDist>maxDist:
                    maxDist = tmpDist
                    cenIdx = j
        clusters[i].append(cenIdx)
        tmpCenter = (tmpCenter*i + mv[clusters[i][0]])/(i+1)

    centers = []
    for i in range(0,k):
        centers.append(mv[clusters[i][0]])

    # Clustering
    for j in range(0,n):
        if j in visited:
            pass
        else:
            minDist = float("inf")
            clsIdx = -1
            for jj in range(len(centers)):
                tmpDist = np.linalg.norm(mv[j]-centers[jj])
                if tmpDist < minDist:
                    minDist = tmpDist
                    clsIdx = jj
            centers[clsIdx] = (centers[clsIdx]*len(clusters[clsIdx]) + mv[j])/(len(clusters[clsIdx])+1)
            clusters[clsIdx].append(j)
    print(clusters)

    color = ['red', 'black', 'blue', 'brown', 'green']
    for i in range(len(clusters)):
        for j in range(0,len(clusters[i])):
            tmpX = mv[clusters[i][j]][0]
            tmpY = mv[clusters[i][j]][1]
            # print(i,j,mv[clusters[i][j]])
            plt.plot(tmpX,tmpY,marker="o",color=color[i])
    plt.show()

if __name__ == '__main__':
    # renameFiles(src,dst)
    sharpImages(dst)
    extractFeatures(imgs)
    clusterImages(4, imgs)


