import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
plt.style.use('ggplot')
# plt.style.use('seaborn-whitegrid')

def print2(*args):
    for arg in args:
        print(arg, end="\n\n")

mydir = "D:\PythonDataScience\MachineLearning\SupervisedME\MarketingME"

# load pickle file
RFMdata = pd.read_pickle(os.path.join(mydir, "RFM.pkl"))

data = RFMdata.loc[:, ["Recency" ,"Frequency",  "MonetaryValue"]]
print2(data.describe(), data.head(), data.shape)

# Remove zero and negative values
data2 = data[data["MonetaryValue"] > 0.00]
print2(data2.head(), data2.shape)

# Log transformation
data = np.log(data2)


scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)


print2(RFMdata, data)

logg = {}

for k in range(1,12):
    kms = KMeans(n_clusters=k)
    kms.fit(data)
    logg[k] = kms.inertia_
    
    
sns.pointplot(x=list(logg.keys()), y=list(logg.values()))
plt.show()

kms = KMeans(n_clusters=4)
kms.fit(data)
data22 = data2.assign(segment = kms.labels_)

print2(data22)


