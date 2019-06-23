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
img = cv2.imread('color22.jpg')  

# converting to gray scale
Img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# remove noise
img = cv2.GaussianBlur(Img_gray,(3,3),0)

# convolute with proper kernels
Laplacian_image = cv2.Laplacian(img,cv2.CV_64F)
Sobel_X_image = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=9) 
Sobel_Y_image = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=9)  


# Plot the images
plt.subplot(221)
plt.imshow(img, cmap = 'gray')
plt.title('Original_image')
plt.axis('off')

plt.subplot(222)
plt.imshow(Laplacian_image, cmap = 'gray')
plt.title('Laplacian_image')
plt.axis('off')

plt.subplot(223)
plt.imshow(Sobel_X_image, cmap = 'gray')
plt.title('Sobel_X_image')
plt.axis('off')

plt.subplot(224),
plt.imshow(Sobel_Y_image, cmap = 'gray')
plt.title('Sobel_Y_image')
plt.axis('off')
plt.show()