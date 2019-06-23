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
imgg = cv2.imread('car22.jpg')  

# converting to gray scale
Img_gray = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)

# remove noise
img = cv2.GaussianBlur(Img_gray,(3,3),0)

# convolute with proper kernels
Laplacian_image = cv2.Laplacian(img,cv2.CV_64F)
Sobel_X_image = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=9) 
Sobel_Y_image = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=9)

t01, t02 = (200, 260)
t11, t12 = (120, 200)
Canny_image = cv2.Canny(img, t01, t02)
Canny_image2 = cv2.Canny(img, t11, t12)

# Plot the images
plt.subplot(231)
plt.imshow(img, cmap = 'gray')
plt.title('Original_image')
plt.axis('off')

plt.subplot(232)
plt.imshow(Laplacian_image, cmap = 'gray')
plt.title('Laplacian_image')
plt.axis('off')

plt.subplot(233)
plt.imshow(Sobel_X_image, cmap = 'gray')
plt.title('Sobel_X_image')
plt.axis('off')

plt.subplot(234),
plt.imshow(Sobel_Y_image, cmap = 'gray')
plt.title('Sobel_Y_image')
plt.axis('off')

plt.subplot(235),
plt.imshow(Canny_image, cmap = 'gray')
plt.title(f'Canny_image {t01}x{t02}')
plt.axis('off')

plt.subplot(236),
plt.imshow(Canny_image2, cmap = 'gray')
plt.title(f'Canny_image2 {t11}x{t12}')
plt.axis('off')
plt.show()

def edge_detectionCam(image, x=None):

    imgg = cv2.imread(image)  

    # converting to gray scale
    Img_gray = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)

    # remove noise
    img = cv2.GaussianBlur(Img_gray,(3,3),0)

    t01, t02 = (200, 260)
    t11, t12 = (120, 200)


    if x == 'canny':
        Canny_image = cv2.Canny(img, t01, t02)
        return Canny_image

    elif x == 'canny2':
        Canny_image2 = cv2.Canny(img, t11, t12)
        return Canny_image2

    elif x == 'sobelx':
        Sobel_X_image = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=9) 
        return Sobel_X_image

    elif x == 'sobely':
        Sobel_Y_image = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=9)
        return Sobel_Y_image

    else:
        Laplacian_image = cv2.Laplacian(img,cv2.CV_64F)
        return Laplacian_image


plt.imshow(edge_detectionCam('car22.jpg', "canny2"), cmap = 'gray')
plt.axis('off')
plt.show()