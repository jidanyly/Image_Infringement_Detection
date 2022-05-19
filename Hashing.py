import imagehash
from PIL import Image
import os
import numpy as np

destpath = 'D:/Programming/AdvancedComputerNetwork/InfringementDetection/data'
imagepath = 'D:/Programming/AdvancedComputerNetwork/InfringementDetection/data/reference'

# Build Phash value Repo
# Using open-source Phash implementation on https://github.com/JohannesBuchner/imagehash
hashes = []
for root, _, files in os.walk(imagepath):
    for i in range(10000):
        file = imagepath + '/' + files[i]
        hashing = imagehash.phash(Image.open(file), hash_size=16).hash.astype(np.uint8)
        hashing = np.array(hashing).flatten()
        hashing = np.packbits(hashing)
        hashes.append(hashing)
np.savez(destpath+'/'+'Phash.npz', hashes)

