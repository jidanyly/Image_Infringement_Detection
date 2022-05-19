import augly.image as imaugs
import os

path = 'D:/Programming/AdvancedComputerNetwork/InfringementDetection/data/reference/'
outpath = 'D:/Programming/AdvancedComputerNetwork/InfringementDetection/data/query/'

# Using open-source image augmentation library on https://github.com/facebookresearch/AugLy
f = []
for root, _, files in os.walk(path):
    for i in range(120):
        f.append(files[i])

for i in range(10):
    #overlay
    imaugs.overlay_emoji(path+f[i*12], opacity=1.0, emoji_size=0.25, output_path=outpath+f[i*12])
    imaugs.overlay_text(path+f[i*12+1], output_path=outpath+f[i*12+1])
    #Color
    imaugs.grayscale(path+f[i*12+2], output_path=outpath+f[i*12+2])
    imaugs.brightness(path+f[i*12+3], output_path=outpath+f[i*12+3], factor=2)
    imaugs.saturation(path+f[i*12+4], output_path=outpath+f[i*12+4],factor=2)
    #Pixel-level transformations
    imaugs.blur(path+f[i*12+5], output_path=outpath+f[i*12+5])
    imaugs.random_noise(path+f[i*12+6], output_path=outpath+f[i*12+6])
    #Spatial transformations
    imaugs.rotate(path+f[i*12+7], output_path=outpath+f[i*12+7], degrees=90)
    imaugs.pad_square(path+f[i*12+8], output_path=outpath+f[i*12+8])
    imaugs.crop(path+f[i*12+9], output_path=outpath+f[i*12+9], x1=0, y1=0, y2=1)
    imaugs.perspective_transform(path+f[i*12+10], sigma=30, output_path=outpath+f[i*12+10])
    imaugs.scale(path+f[i*12+11], factor=0.6, output_path=outpath+f[i*12+11])