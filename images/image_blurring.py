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

kernel12 = np.float32([[-1, -1, -1],
                      [-1, 9, -1],
                      [-1, -1, -1]])

sharpen_image2 = cv2.filter2D(img23, -1, kernel12)

cv2.imshow('sharpen_image', sharpen_image2)
cv2.waitKey()
cv2.destroyAllWindows()


# Blurring images
blur_kernel = np.ones((3, 3))
blur_image = cv2.filter2D(img23, -1, blur_kernel)

cv2.imshow('blur_image', blur_image)
cv2.waitKey()
cv2.destroyAllWindows()

# Blurring images with normalization
blur_kernel2 = np.ones((3, 3)) * 1/9
blur_image2 = cv2.filter2D(img23, -1, blur_kernel2)

cv2.imshow('blur_image_normalized', blur_image2)
cv2.waitKey()
cv2.destroyAllWindows()

# Blurring images with more powerful normalization
blur_kernel3 = np.ones((10, 10)) * 1/100
blur_image22 = cv2.filter2D(img23, -1, blur_kernel3)

cv2.imshow('blur_image_Powerful', blur_image22)
cv2.waitKey()
cv2.destroyAllWindows()
