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

print2( wt, ko)
data = pd.DataFrame(columns=[*wt ,*ko], index=genes)
print2(data.head())

n = 2
for gene in data.index:
    data.loc[gene, :"wt6"] =  np.random.poisson(lam=np.random.randint(10,100),size=6)
    np.random.seed(90 + n)
    data.loc[gene, "ko1":] =  np.random.poisson(lam=np.random.randint(10,100),size=6)
    n += 5
    
    
