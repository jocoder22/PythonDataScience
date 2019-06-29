#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import cv2
from skimage import io, exposure
from skimage.feature import hog

sp = '\n\n'
# url = 'https://live.staticflickr.com/7856/47133183871_52540009ac_b.jpg'
url = 'https://live.staticflickr.com/3900/14288309597_d7bfd2bab6_b.jpg'

# imgg = cv2.imread('car22.jpg') 
imgg = io.imread(url)

# converting to gray scale
Img_gray = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)

# remove noise
img = cv2.GaussianBlur(Img_gray,(7,7),0)

# cv2.imshow('self_drawing_car', img)
cv2.imshow('Walgreens Truck', img)
cv2.waitKey()
cv2.destroyAllWindows()