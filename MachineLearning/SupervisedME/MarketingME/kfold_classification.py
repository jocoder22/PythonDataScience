import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
# plt.style.use('ggplot')
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
data = data[data["MonetaryValue"] > 0.00]
print2(data.head(), data.shape)

# Log transformation
data = np.log(data)


scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)


print2(RFMdata, data)