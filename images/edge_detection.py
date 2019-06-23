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
img2 = cv2.imread('car22.jpg', 0)  # load as gray directly

# Sobel detection
x_sobel = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=7)
cv2.imshow('x_sobel', x_sobel)
cv2.waitKey()
cv2.destroyAllWindows()