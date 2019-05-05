import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

src = "data/"
dst = "data_processed/"

imgs = []

for filename in os.listdir(src):
    tmp = cv2.imread(src+filename,0)
    # tmp = cv2.imread(src+filename)
    # res = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
    res = cv2.Sobel(tmp, cv2.CV_64F,1,1, ksize=3)
    # res = cv2.Laplacian(tmp,cv2.CV_64F,ksize=5)
    # res = cv2.Scharr(tmp,cv2.CV_64F,1,0)
    imgs.append(res)

means = []
varis = []
for img in imgs:
    # means.append(np.mean(img))
    # means.append(np.sum(img**2)/img.size)
    means.append(np.sum(np.absolute(img))/img.size)
    # varis.append(np.var(img))
    varis.append(np.std(img))
    plt.plot(means,varis,'ro')
    # plt.scatter(means, varis)

print(imgs[-1],imgs[-2])
plt.show()
