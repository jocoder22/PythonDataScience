#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import seaborn as sns
from printdescribe import print2

genes = ["gene"+str(i) for i in range(1,101)]
wt = ["wt"+str(i) for i in range(1,7)]
ko = ["ko"+str(i) for i in range(1,7)]
data = pd.DataFrame(columns=[*wt ,*ko], index=genes)

print2(wt, ko, data.head())

for gene in data.index:
  data.loc[gene, :] = np.random.poisson(lam=np.random.randint(10,200),size=data.shape[1])

print2(data.head())
