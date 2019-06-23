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
img = plt.imread('car22.jpg')
plt.imshow(img)
plt.show()

# load image
img22 = cv2.imread('car22.jpg')
img23 = cv2.imread('car22.jpg', 0)  # load as gray directly
img2 = cv2.cvtColor(img22, cv2.COLOR_BGR2GRAY)
img3 = cv2.cvtColor(img22, cv2.COLOR_BGR2RGB)
height, width = img.shape[:2]
cv2.imshow('lamborghini', img22)
cv2.imshow('lamborghini_GRAY', img2)
cv2.imshow('lamborghini_RGB', img3)
cv2.imshow('lamborghini_GRAY_direct', img23)
cv2.waitKey()
cv2.destroyAllWindows()


# Sharpen image
kernel1 = np.float32([[0, -1, 0],
                      [-1, 5, -1],
                      [0, -1, 0]])

sharpen_image = cv2.filter2D(img23, -1, kernel1)

cv2.imshow('sharpen_image', sharpen_image)
cv2.waitKey()
cv2.destroyAllWindows()