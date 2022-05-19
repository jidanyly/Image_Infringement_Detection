import faiss
import numpy as np
import imagehash
from PIL import Image
import cv2 as cv
from matplotlib import pyplot as plt
import time
import os

# Using FLANN built in OpenCV for ORB feature matching
# Following the standard pipeline for ORB feature matching https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
# Using open-source ORB implementation by OpenCV
# Using Faiss by Facebook AI for Phash value matching on https://github.com/facebookresearch/faiss

destpath = 'D:/Programming/AdvancedComputerNetwork/InfringementDetection/data/'
imagepath = 'D:/Programming/AdvancedComputerNetwork/InfringementDetection/data/query/'
nb = np.load(destpath + '/' + 'Phash.npz')
hash_repo = np.array(nb['arr_0'])
nb = np.load(destpath + '/' + 'ORB.npz', allow_pickle=True)
orb_repo = nb['arr_0']
orb = cv.ORB_create(nfeatures=40)
right = 0


def detection(file):
    img = cv.imread(imagepath + file, 0)
    nq = imagehash.phash(Image.open(imagepath + file), hash_size=16).hash
    nq = np.array([np.packbits(np.array(nq).flatten())])
    index = faiss.IndexBinaryFlat(256)
    index.add(hash_repo)
    k = 1
    D, I = index.search(nq, k)
    lor = '_r' if I[0][0] % 2 else '_l'
    pred = str(int(I[0][0] / 2)).zfill(5) + lor + '.jpg'
    if D[0][0] > 80:
        index_params = dict(algorithm=6,
                            table_number=6,  # 12
                            key_size=12,  # 20
                            multi_probe_level=1)  # 2
        search_params = dict(checks=50)  # or pass empty dictionary
        kp1, des1 = orb.detectAndCompute(img, None)
        result = []
        matchnum = 0
        similar = 0
        i = 0
        for des in orb_repo:
            if des is None:
                i += 1
                continue
            flann = cv.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(des1, des, k=2)
            good = []
            for match in matches:
                if len(match) != 2:
                    continue
                m, n = match
                if m.distance < 0.7 * n.distance:
                    good.append([m])
            result.append(len(matches))
            if len(good) > matchnum:
                matchnum = len(good)
                similar = i
            i += 1
        lor = '_r' if similar % 2 else '_l'
        pred = str(int(similar / 2)).zfill(5) + lor + '.jpg'
        if pred == file:
            return 1
        else:
            return 0
    else:
        if pred == file:
            return 1
        else:
            return 0


for root, _, files in os.walk(imagepath):
    start = time.time()
    for i in range(120):
        right += detection(files[i])
    end = time.time()
    print(right)
    print(end - start)
