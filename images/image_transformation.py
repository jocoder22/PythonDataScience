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
img = cv2.resize(meds,(1200,700))
cv2.imshow('mercedes', img)
cv2.waitKey()
cv2.destroyAllWindows()