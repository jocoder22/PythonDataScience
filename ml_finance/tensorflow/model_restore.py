#!/usr/bin/env python
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import shutil
# import tensorflow as tf

from ipywidgets import interact


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from printdescribe import print2, changepath
from datetime import datetime
print2(" ")

path22 = r"D:\PythonDataScience"
sys.path.insert(0, path22)
import input_data