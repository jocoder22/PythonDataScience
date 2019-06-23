#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import cv2
from skimage import io

path = r'C:\Users\Jose\Desktop\PythonDataScience\images'
os.chdir(path)

# load image
img4 = plt.imread('car22.jpg')
plt.imshow(img4)
plt.show()

# load image
img = cv2.imread('car22.jpg')
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img3 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
height, width = img.shape[:2]
cv2.imshow('lamborghini', img)
cv2.imshow('lamborghini_GRAY', img2)
cv2.imshow('lamborghini_RGB', img3)
cv2.waitKey()
cv2.destroyAllWindows()

