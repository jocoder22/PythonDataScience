#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import cv2
from skimage import io, exposure
from skimage.feature import hog

def print2(*args):
    for arg in args:
        print(arg, end='\n\n')
  
sp = {"sep":"\n\n", "end":"\n\n"} 

path = r'C:\Users\Jose\Desktop\Holder'
os.chdir(path)
sp = '\n\n'

# url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTEc92kYxyNsx6ZxWYF6KJJz-QZWUj0jXBleB2tEg6yBekggb28'
# url = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSESxr13ODvh5lfb1TxT8LgDbzWP44sD5n1z_Nf-697su_jona3zw'

# load image
imgg = cv2.imread('car22.jpg') 
# imgg = io.imread(url)

# converting to gray scale
Img_gray = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)

# remove noise
img = cv2.GaussianBlur(Img_gray,(7,7),0)

img_c = cv2.cornerHarris(img, 3, 5, 0.1)
img_dilate = cv2.dilate(img_c, np.ones((5, 5), np.uint8), iterations=1)


print(img_dilate.max(), **sp)
Img_gray2 = imgg.copy()
Img_gray2[img_dilate > 0.02 * img_dilate.max()] = [255, 0, 0]

cv2.imshow('lamborghini_with_Corners', Img_gray2)
cv2.waitKey()
cv2.destroyAllWindows()

plt.imshow(Img_gray2)
plt.axis('off')
plt.show()


features, hog_img = hog(Img_gray,visualize=True, 
                pixels_per_cell=(9, 9), cells_per_block=(2, 2))
img_hog = exposure.rescale_intensity(hog_img, in_range=(0, 2))
plt.imshow(img_hog)
plt.axis('off')
plt.show()