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

# load image
meds = cv2.imread('car22.jpg')
img = cv2.resize(meds,(1250,650))
height, width = img.shape[:2]
cv2.imshow('lamborghini', img)
cv2.waitKey()
cv2.destroyAllWindows()