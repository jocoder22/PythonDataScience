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
cv2.imshow('lamborghini', img)
cv2.waitKey()
cv2.destroyAllWindows()

# perform rotation
height, width = img.shape[:2]
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