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

url1 = 'https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTdYoq9p_F3wQRMFzCN60eCXJhSfPq4daFMQqV4zVhK6f_IGye1'
sp = '\n\n'



# img = mpimg.imread(url, format='jpeg')
img = io.imread(url1)
plt.axis('off')
plt.imshow(img)
plt.show()