#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import cv2
from skimage import io

path = r'C:\Users\Jose\Desktop\Holder'
os.chdir(path)
sp = '\n\n'

url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTEc92kYxyNsx6ZxWYF6KJJz-QZWUj0jXBleB2tEg6yBekggb28'

# load image
# imgg = cv2.imread('car22.jpg') 
imgg = io.imread(url)

# converting to gray scale
Img_gray = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)

# remove noise
img = cv2.GaussianBlur(Img_gray,(7,7),0)

img_c = cv2.cornerHarris(img, 3, 5, 0.1)
img_dilate = cv2.dilate(img_c, np.ones((5, 5), np.uint8), iterations=1)


print(img_dilate.max(), end=sp)
Img_gray2 = imgg.copy()
Img_gray2[img_dilate > 0.02 * img_dilate.max()] = [255, 0, 0]

cv2.imshow('lamborghini_with_Corners', Img_gray2)
cv2.waitKey()
cv2.destroyAllWindows()

plt.imshow(Img_gray2)
plt.axis('off')
plt.show()