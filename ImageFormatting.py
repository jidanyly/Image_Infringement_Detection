import augly
import os
import shutil

# in total 10000 reference images, where 5000 in left, 5000 in right
filepathl = 'D:/Programming/AdvancedComputerNetwork/Data/left'
filepathr = 'D:/Programming/AdvancedComputerNetwork/Data/right'
destpath = 'D:/Programming/AdvancedComputerNetwork/InfringementDetection/data/reference'
for root, _, files in os.walk(filepathl):
    for i in range(5000):
        file = files[i]
        shutil.copy(filepathl+'/'+file, destpath+'/'+file[0:5]+'_l.jpg')

for root, _, files in os.walk(filepathr):
    for i in range(5000):
        file = files[i]
        shutil.copy(filepathr+'/'+file, destpath+'/'+file[0:5]+'_r.jpg')

