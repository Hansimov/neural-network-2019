import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import random
from shutil import copyfile

src = "data/"
dst = "data_renamed/"

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

if __name__ == '__main__':
    renameFiles()
