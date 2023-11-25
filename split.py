import shutil 
import os 

path = "./images"
dst_dir = "./input/"
for file in os.listdir(path):
    name, _ = os.path.splitext(file)
    if name[-1] == '0':
        destination = dst_dir + file
        location = path + '/' + file
        shutil.move(location,destination)