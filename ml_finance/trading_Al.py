#Import the necessary libraries
import pandas as pd
import pandas_datareader.data as dr
import matplotlib
import matplotlib.pyplot as plt
import math
import statsmodels.tsa.stattools as ts
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from printdescribe import print2

start_date = '2013-01-01'
end_date = '2020-02-29'
# end_date = '2020-09-11'

assets = 'FDN'
datasets = dr.DataReader(assets, data_source='yahoo', start = start_date)
print2(datasets.tail())


datasets['Adj Close'].plot(figsize=(10,5))
plt.show();
