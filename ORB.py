import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import time
import os
import faiss

destpath = 'D:/Programming/AdvancedComputerNetwork/InfringementDetection/data'
imagepath = 'D:/Programming/AdvancedComputerNetwork/InfringementDetection/data/reference'
orb = cv.ORB_create(nfeatures=40)

# Build ORB features Repo
# Using open-source ORB implementation by OpenCV
orbs = []
for root, _, files in os.walk(imagepath):
    for i in range(10000):
        file = imagepath + '/' + files[i]
        img = cv.imread(file, 0)
        kp1, des1 = orb.detectAndCompute(img, None)
        orbs.append(des1)
np.savez(destpath+'/'+'ORB.npz', orbs)