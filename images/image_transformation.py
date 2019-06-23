#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from skimage import io

path = r'C:\Users\Jose\Desktop\PythonDataScience\images'
os.chdir(path)

url = 'https://timedotcom.files.wordpress.com/2015/01/the-mercedes-benz-f-015.jpg'

# # read the image
# mmm = io.imread(url)
# plt.imshow(mmm)
# # plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
# plt.show()

meds = cv2.imread('car22.jpg')
img = cv2.resize(meds,(1250,650))
height, width = img.shape[:2]
cv2.imshow('lamborghini', img)
cv2.waitKey()
cv2.destroyAllWindows()



# perform rotation
rot = cv2.getRotationMatrix2D((width/2, height/2), 90, 0.4)
image_r = cv2.warpAffine(img, rot,(width, height))
cv2.imshow('lamborghini_rotated_45', image_r)
cv2.waitKey()
cv2.destroyAllWindows()


# turn image upside down
updown = img[::-1]
invert_mirror = img[::-1,::-1,:]
mirror_image = img[:,::-1,:]
# mirror_image = img[::-1,::-1,::-1]


plt.subplot(221)
plt.imshow(img)
plt.title('original')
plt.axis('off')

plt.subplot(223)
plt.imshow(updown)
plt.axis('off')
plt.title('upsidedown')

plt.subplot(224)
plt.imshow(invert_mirror)
plt.axis('off')
plt.title('invert_mirror')

plt.subplot(222)
plt.imshow(mirror_image)
plt.axis('off')
plt.title('mirror_image')
plt.show()

# shifting image
shift_matrix = np.float32([[1, 0, -100],   # shifting left(minus) or right
                           [0, 1, 0]])    # shifting up(minus) or down

shift_right = cv2.warpAffine(img, shift_matrix, (width, height))
cv2.imshow('lamborghini_shifted_right', shift_right)
cv2.waitKey()
cv2.destroyAllWindows()

# resizing
resize_img = cv2.resize(img, None, fx=1.7, fy=1.5, interpolation=cv2.INTER_LINEAR) # Enlarge, fast, good
resize_img2 = cv2.resize(img, None, fx=1.7, fy=1.5, interpolation=cv2.INTER_CUBIC)  # Enlarge, slow but better
resize_img3 = cv2.resize(img, None, fx=0.7, fy=0.5, interpolation=cv2.INTER_AREA) # Shrink
cv2.imshow('lamborghini_resized', resize_img)
cv2.imshow('lamborghini_resized2', resize_img2)
cv2.imshow('lamborghini_resized3', resize_img3)
cv2.waitKey()
cv2.destroyAllWindows()


# plt.imread(img)
plt.imshow(img)
plt.show()

# x0, x1, x2, x3 = 100, 100, 450, 450
# y0, y1, y2, y3 = 0, 400, 400, 0

x0, x1, x2, x3 =  124, 80, 205, 230
y0, y1, y2, y3 = 228, 306, 319, 246

x0, x1, x2, x3 = 230, 308, 326, 265
y0, y1, y2, y3 = 112, 59, 193, 239

xx = 400
yy = 600

origin_matrix = np.float32([[x0, y0],[x1 ,y1],
                          [x2, y2], [x3, y3]])

dest_matrix = np.float32([[0, 0], [xx, 0],
                          [xx, yy], [0, yy]])

pers = cv2.getPerspectiveTransform(origin_matrix, dest_matrix)

tires = cv2.warpPerspective(img, pers, (xx, yy))
cv2.imshow('lamborghini_tire', tires)
cv2.waitKey()
cv2.destroyAllWindows()


x0, x1, x2, x3 = 200, 450, 520, 170
y0, y1, y2, y3 = 60, 150, 500, 470
xx, yy = 800, 600

origin_matrix = np.float32([[x0, y0],[x1 ,y1],
                          [x2, y2], [x3, y3]])

dest_matrix = np.float32([[0, 0], [xx, 0],
                          [xx, yy], [0, yy]])

pers = cv2.getPerspectiveTransform(origin_matrix, dest_matrix)

tires = cv2.warpPerspective(img, pers, (xx, yy))
cv2.imshow('lamborghini_tire', tires)
cv2.waitKey()
cv2.destroyAllWindows()


url2 = 'https://www.here.com/sites/g/files/odxslz166/files/styles/hero_banner_secondary_xs/public/2019-03/HERE_Road_Signs-product_detail-hero_2880x1440.jpg?itok=oDgCaLLg'
mmm = io.imread(url2)
plt.imshow(mmm)
# plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

cv2.imshow('Speed Limit', mmm)
cv2.waitKey()
cv2.destroyAllWindows()


x0, x1, x2, x3 = 385, 473, 476, 388
y0, y1, y2, y3 = 69, 46, 183, 192
xx, yy = 800, 600

origin_matrix = np.float32([[x0, y0],[x1 ,y1],
                          [x2, y2], [x3, y3]])

dest_matrix = np.float32([[0, 0], [xx, 0],
                          [xx, yy], [0, yy]])

pers = cv2.getPerspectiveTransform(origin_matrix, dest_matrix)

road_sign = cv2.warpPerspective(mmm, pers, (xx, yy))
cv2.imshow('Speed Limit', road_sign)
cv2.waitKey()
cv2.destroyAllWindows()