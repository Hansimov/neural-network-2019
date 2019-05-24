import cv2

root = "./data_renamed/"
# img = cv2.imread(root+"A1.bmp",0)
# print(img.shape)
# imgx = cv2.resize(img,(100,100))
# print(imgx.shape)
# cv2.imwrite("A1x.bmp",imgx)

img1 = cv2.imread(root+"Dtimg.jpg",0)
print(img1.shape)
img1x = cv2.resize(img1, (100,100))
print(img1x.shape)
cv2.imwrite("Dtimgx.jpg",img1x)
