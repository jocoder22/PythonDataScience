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

url1 = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTdYoq9p_F3wQRMFzCN60eCXJhSfPq4daFMQqV4zVhK6f_IGye1'
sp = '\n\n'



# img = mpimg.imread(url, format='jpeg')
img = io.imread(url1)
plt.axis('off')
plt.imshow(img)
plt.show()

print(img.shape, img.size, sep=sp, end=sp)

print(list(range(1,5)))

# Create gray image
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.imshow(img_gray, cmap='gray')
plt.axis('off')
plt.show()
print(img_gray.shape, img_gray.size, sep=sp, end=sp)

# save the image
# cv2.imwrite('gray_image.jpg', img_gray)
plt.imsave('color22.jpg', img)


# detect white line in gray image
gray2 = img_gray.copy()
gray2[gray2 < 230] = 0
plt.imshow(gray2, cmap='gray')
plt.axis('off')
plt.show()



# detect white line in color image
color = img.copy()
color[(color[..., 0] < 230) | (color[..., 1] < 230) | (color[..., 2] < 230)] = 0
plt.imshow(color)
plt.axis('off')
plt.show()




url2 = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcT9mzaXhllnV5VHCEUp9zuqbEoRbe_OihMvPJ4TBxoeyClofw4x'
url3 = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTNiMDQkAVuH2P-OPl3ZUaQQW369h4ZquHxeLdeRht13DDBEf27'
url4 = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcREko7eh7dMSqoee1iz75W1NAP44D85OLSPkwc_6pErAqlRpOK4CA'
url5 = 'https://timedotcom.files.wordpress.com/2015/01/the-mercedes-benz-f-015.jpg'
url6 = 'https://www.mercedes-benz.com/wp-content/uploads/sites/3/2015/01/10-Mercedes-Benz-F-015-Luxury-in-Motion-1180x436.jpg'


# read the image
meds = cv2.imread('car22.jpg')
img4 = cv2.resize(meds,(860,980))
cv2.imshow('mercedes', img4)
cv2.waitKey(0)
cv2.destroyAllWindows()

# show with matplotlib
plt.axis('off')
plt.imshow(meds)
plt.show()


# load the images
mp = io.imread(url4)
plt.imshow(mp)
plt.axis('off')
plt.show()
R, G, B = mp[110, 234]
print(R,G,B)


lkk = [url1, url2, url3, url4, url5, url6]
for i in range(0, 6):
    lk = lkk[i]
    mmm = io.imread(lk)
    plt.imshow(mmm)
    plt.title(f'Image {i+1}')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()



# Create HSV image
imgg = cv2.imread('color22.jpg')
cv2.imshow('Original Image', mp)
img_hsv = cv2.cvtColor(mp, cv2.COLOR_BGR2HSV)
cv2.imshow('HSV Image', img_hsv)
cv2.imshow('Hue Channel', img_hsv[..., 0])
cv2.imshow('Value Channel', img_hsv[..., 1])
cv2.imshow('Saturation Channel', img_hsv[..., 2])
cv2.waitKey()
cv2.destroyAllWindows()