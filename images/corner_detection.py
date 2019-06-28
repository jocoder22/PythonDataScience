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


# load image
imgg = cv2.imread('car22.jpg')  

# converting to gray scale
Img_gray = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)