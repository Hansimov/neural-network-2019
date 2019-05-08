import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
from shutil import copyfile

src = "data/"
dst = "data_renamed/"
tgt = "data_processed/"

def renameFiles():
    if not os.path.exists(dst): 
        os.mkdir(dst)
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
def sharpImages():
    if not os.path.exists(tgt): 
        os.mkdir(tgt)
    global imgs
    for filename in os.listdir(dst):
        tmp = cv2.imread(dst+filename,0)
        # res = tmp
        # tmp = cv2.imread(src+filename)
        # res = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
        # res = cv2.Sobel(tmp, cv2.CV_64F,1,1, ksize=5)
        res = cv2.Laplacian(tmp,cv2.CV_64F,ksize=5)
        # res = cv2.Scharr(tmp,cv2.CV_64F,1,0)
        imgs.append([res,filename])
        cv2.imwrite(tgt+filename,res)

means = []
varis = []
names = []
colors = ['red', 'green', 'blue', 'black', 'cyan']
types = ['A','B','C','D','E']
def extractFeatures():
    global imgs
    imgs = []
    for filename in os.listdir(tgt):
        pix = cv2.imread(tgt+filename)
        imgs.append([pix,filename])
    global means, varis, names

    plt.figure(0)
    for img in imgs:
        pix = img[0]
        # means.append(np.mean(pix))
        # means.append(np.sum(pix**2)/img.size)
        means.append(np.sum(np.absolute(pix))/pix.size)
        # means.append(np.sum(pix)/pix.size)
        # varis.append(np.var(pix))
        varis.append(np.std(pix))
        names.append(img[1])
        # plt.scatter(means, varis)
        # print()

        filename = img[1]
        plt.plot(means[-1],varis[-1], marker='o',color=colors[types.index(filename[0])])

    means = np.array(means)
    varis = np.array(varis)
    [means, varis] = list(map(lambda x: np.interp(x, (x.min(),x.max()),(0,10)), [means, varis]))

    # plt.plot(means, varis, 'ro')
    # plt.show()

def clusterImages(k):
    n = len(means)
    mv = []
    for idx in range(0,n):
        mv.append(np.array([means[idx],varis[idx]]))

    visited = []
    clusters = [[] for _ in range(k)]

    ## Initialize centers

    # centerIdxs = random.sample(range(n),k)
    # visited  = visited + centerIdxs
    # for i in range(k):
    #     clusters[i].append(centerIdxs[i])

    clusters[0].append(random.randint(0,n-1))
    visited.append(clusters[0][0])
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
        visited.append(cenIdx)
        tmpCenter = (tmpCenter*i + mv[clusters[i][0]])/(i+1)


    centers = []
    centersOld = []
    for i in range(0,k):
        centers.append(mv[clusters[i][0]])

    #$ Clustering
    isClustered = False
    epoch = 0
    # while (not isClustered) and (epoch<10):
    while not isClustered:
        if epoch==0:
            pass
        else:
            visited = []
            clusters = [[] for _ in range(k)]

        print(epoch)
        epoch += 1

        for j in range(0,n):
            if j in visited:
                pass
            else:
                minDist = float("inf")
                clsIdx = -1
                for i in range(k):
                    tmpDist = np.linalg.norm(mv[j]-centers[i])
                    if tmpDist < minDist:
                        minDist = tmpDist
                        clsIdx = i
                clusters[clsIdx].append(j)

        ## Check is clustered
        if len(centersOld) == 0:
            centersOld = centers.copy()
            continue
        else:
            isClustered = True
            for i in range(0,k):
                centerTmp = np.array([0,0])
                for idx in clusters[i]:
                    centerTmp = centerTmp + mv[idx]
                centers[i] = centerTmp/len(clusters[i])
                isClustered = isClustered and (centers[i][0]==centersOld[i][0]) and (centers[i][1]==centersOld[i][1])
            print(centers[0],centersOld[0],isClustered)
            centersOld = centers.copy()

    # print(sum(list(map(lambda x: len(clusters[x]), [0,1,2,3]))))
    # print(list(map(lambda x: len(clusters[x]), [0,1,2,3])))

    plt.figure(1)
    for i in range(len(clusters)):
        for j in range(0,len(clusters[i])):
            tmpX = mv[clusters[i][j]][0]
            tmpY = mv[clusters[i][j]][1]
            # print(i,j,mv[clusters[i][j]])
            plt.plot(tmpX,tmpY,marker="o",color=colors[i])
    plt.show()

if __name__ == '__main__':
    # renameFiles()
    # sharpImages()
    extractFeatures()
    clusterImages(4)


