import os
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
# plt.style.use('ggplot')
# plt.style.use('seaborn-whitegrid')

def print2(*args):
    for arg in args:
        print(arg, end="\n\n")

mydir = "D:\PythonDataScience\MachineLearning\SupervisedME\MarketingME"

# load pickle file
RFMdata = pd.read_pickle(os.path.join(mydir, "RFM.pkl"))

data = RFMdata.loc[:, ["Recency" ,"Frequency",  "MonetaryValue"]]

print2(RFMdata, data)