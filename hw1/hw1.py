import cv2
import numpy as np
from matplotlib import pyplot as plt

src = "data/"
dst = "data_processed/"

img1 = cv2.imread(src+"1.bmp",0)
img2 = cv2.imread(src+"2dim100.bmp",0)
img3 = cv2.imread(src+"1015.png",0)
img4 = cv2.imread(src+"1301.jpg",0)
img5 = cv2.imread(src+"timg.jpg",0)

lap1 = cv2.Laplacian(img1,cv2.CV_64F)
lap2 = cv2.Laplacian(img2,cv2.CV_64F)
lap3 = cv2.Laplacian(img3,cv2.CV_64F)
lap4 = cv2.Laplacian(img4,cv2.CV_64F)
lap5 = cv2.Laplacian(img5,cv2.CV_64F)

# lap1 = cv2.Scharr(img1,cv2.CV_64F,1,0)
# lap2 = cv2.Scharr(img2,cv2.CV_64F,1,0)
# lap3 = cv2.Scharr(img3,cv2.CV_64F,1,0)
# lap4 = cv2.Scharr(img4,cv2.CV_64F,1,0)
# lap5 = cv2.Scharr(img5,cv2.CV_64F,1,0)

# lap1 = cv2.Sobel(img1,cv2.CV_64F,0,1,ksize=5)
# lap2 = cv2.Sobel(img2,cv2.CV_64F,0,1,ksize=5)
# lap3 = cv2.Sobel(img3,cv2.CV_64F,0,1,ksize=5)
# lap4 = cv2.Sobel(img4,cv2.CV_64F,0,1,ksize=5)
# lap5 = cv2.Sobel(img5,cv2.CV_64F,0,1,ksize=5)

print(list(map(lambda x:round(np.mean(abs(x)),1),[lap1,lap2,lap3,lap4,lap5])))
print(list(map(lambda x:round(np.var(x),1),[lap1,lap2,lap3,lap4,lap5])))

row,col = 2,6

plt.subplot(row,col,1),plt.imshow(img1,cmap = 'gray')
# plt.title('1.bmp'), plt.xticks([]), plt.yticks([])
plt.subplot(row,col,2),plt.imshow(lap1,cmap = 'gray')
# plt.title('1_lap.bmp'), plt.xticks([]), plt.yticks([])

plt.subplot(row,col,3),plt.imshow(img2,cmap = 'gray')
# plt.title('2dim100.bmp'), plt.xticks([]), plt.yticks([])
plt.subplot(row,col,4),plt.imshow(lap2,cmap = 'gray')
# plt.title('2dim100_lap.bmp'), plt.xticks([]), plt.yticks([])

plt.subplot(row,col,5),plt.imshow(img3,cmap = 'gray')
# plt.title('1015.png'), plt.xticks([]), plt.yticks([])
plt.subplot(row,col,6),plt.imshow(lap3,cmap = 'gray')
# plt.title('1015_lap.png'), plt.xticks([]), plt.yticks([])

plt.subplot(row,col,7),plt.imshow(img4,cmap = 'gray')
# plt.title('1301.jpg'), plt.xticks([]), plt.yticks([])
plt.subplot(row,col,8),plt.imshow(lap4,cmap = 'gray')
# plt.title('1301_lap.jpg'), plt.xticks([]), plt.yticks([])

plt.subplot(row,col,9),plt.imshow(img5,cmap = 'gray')
# plt.title('timg.jpg'), plt.xticks([]), plt.yticks([])
plt.subplot(row,col,10),plt.imshow(lap5,cmap = 'gray')
# plt.title('timg_lap.jpg'), plt.xticks([]), plt.yticks([])

# plt.show()