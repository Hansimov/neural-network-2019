import os

cmd = "D:/ImageMagick/convert.exe "

for filename in os.listdir("./"):
    name, ext = os.path.splitext(filename)
    if ext == ".bmp":
        os.system(cmd+filename+" "+name+".png")