import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
plt.style.use('ggplot')


df = pd.read_csv('power.csv', sep=',')

# this finds the columns that are binary
[s for s in df if df[s].max() == 1]


# thi finds the total number that is binary
(df.max() == 1).sum()
len([s for s in df if df[s].max() == 1])

# this finds the columns that are of type float64
[s for s in df if df[s].dtype == 'float64']