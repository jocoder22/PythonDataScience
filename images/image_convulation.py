#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from skimage import io

url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTdYoq9p_F3wQRMFzCN60eCXJhSfPq4daFMQqV4zVhK6f_IGye1'
sp = '\n\n'

# img = mpimg.imread(url, format='jpeg')
img = io.imread(url)
plt.axis('off')
plt.imshow(img)
plt.show()

print(img.shape, img.size, sep=sp, end=sp)


# Create gray image
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.imshow(img_gray, cmap='gray')
plt.axis('off')
plt.show()
print(img_gray.shape, img_gray.size, sep=sp, end=sp)
