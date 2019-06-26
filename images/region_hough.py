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

url1 = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTdYoq9p_F3wQRMFzCN60eCXJhSfPq4daFMQqV4zVhK6f_IGye1'
sp = '\n\n'



# img = mpimg.imread(url, format='jpeg')
img = io.imread(url1)
# plt.axis('off')
# plt.imshow(img)
# plt.show()
# print(img.shape, end=sp)

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('gray_img', gray_img)
# cv2.waitKey()
# cv2.destroyAllWindows()
# print(gray_img.shape, end=sp)


# select point of interest
x0, x1, x2, x3, x4 = 0, 70, 140, 290, 0
y0, y1, y2, y3, y4 = 110, 60, 60, 160, 160


region_matrix = np.array([[(x0, y0),(x1 ,y1), 
                          (x2, y2),(x3, y3),(x4, y4)]], dtype=np.int32)
blanks = np.zeros_like(gray_img)

mask = cv2.fillPoly(blanks, region_matrix, 255)

region_img = cv2.bitwise_and(gray_img, mask)

cv2.imshow('region_image', region_img)
cv2.waitKey()
cv2.destroyAllWindows()
print(region_img.shape, end=sp)

# Hough transformation
# remove noise
img = cv2.GaussianBlur(gray_img, (7, 7),0)

# define threshold
t11, t12 = (20, 200)

Canny_image = cv2.Canny(img, t11, t12)

# Define region of Interest
region_img2 = cv2.bitwise_and(Canny_image, mask)

# show the image
cv2.imshow('region_image', region_img2)
cv2.waitKey()
cv2.destroyAllWindows()
print(region_img.shape, end=sp)

rho = 2
theta = np.pi/180
th0 = 20
line_l = 10
line_gap = 5

houg_img = cv2.HoughLinesP(region_img2,rho, theta, th0, 
                np.array([]),minLineLength=line_l, maxLineGap=line_gap)

blanks = np.zeros_like(img)

for lines in houg_img:
    for x0,y0,x1,y1 in lines:
        cv2.line(blanks,(x0, y0), (x1, y1), (255,0, 0), 5)

cv2.imshow('blanks', blanks)
cv2.waitKey()
cv2.destroyAllWindows()
print(blanks.shape, end=sp)

alpha, beta, gamma = 1, 1, 0

line_image = cv2.addWeighted(img, alpha, blanks, beta, gamma)



cv2.imshow('line_image', line_image)
cv2.waitKey()
cv2.destroyAllWindows()
print(line_image.shape, end=sp)

plt.imshow(line_image)
plt.show()