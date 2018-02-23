
from os import listdir
from os.path import isfile, join
import os
mypath = 'C:\\Users\\foamliu\\faceswap\\faceswap\\data\\liuyang'
onlyfiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]

index = 0
for f in onlyfiles:
    print(f)
    newname = join(mypath, str(index) + '.jpg')
    print(newname)
    index = index + 1
    os.rename(f, newname)

