#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = r"C:\Users\Jose\Pictures"
os.chdir(path)

img = plt.imread('pic11.png')
plt.axis('off')
plt.imshow(img)
plt.show()

img.shape

# show the three channels
for i in range(3):
    onechannel = img[..., i]
    plt.imshow(onechannel)
    plt.axis('off')
    plt.pause(2)
    plt.clf()

plt.close()

# reverse the channels
rev_color = img[..., ::-1]
plt.subplot(121)
plt.imshow(img)
plt.axis('off')
plt.subplot(122)
plt.imshow(rev_color)
plt.axis('off')
plt.show()

# turn image upside down
updown = img[::-1]
invert_mirror = img[::-1,::-1,:]
mirror_image = img[:,::-1,:]


plt.subplot(141)
plt.imshow(img)
plt.title('original')
plt.axis('off')

plt.subplot(142)
plt.imshow(updown)
plt.axis('off')
plt.title('upsidedown')

plt.subplot(143)
plt.imshow(invert_mirror)
plt.axis('off')
plt.title('invert_mirror')

plt.subplot(144)
plt.imshow(mirror_image)
plt.axis('off')
plt.title('mirror_image')
plt.show()

