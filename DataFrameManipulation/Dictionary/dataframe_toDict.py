#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_datareader as pdr
from datetime import date
from printdescribe import changepath
import time


path = "E:\PythonDataScience\ensemble"

with changepath(path):
    data = pd.read_csv("lifeExp.csv")

print(data.head())