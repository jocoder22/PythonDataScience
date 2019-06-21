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

# save the image
# cv2.imwrite('gray_image.jpg', img_gray)
# plt.imsave('gray_image2.jpg', img_gray)

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

