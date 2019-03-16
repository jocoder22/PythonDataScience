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