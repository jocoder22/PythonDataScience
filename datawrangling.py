import os
import pandas as pd
import matplotlib.pyplot as plt
from pandas import Series, DataFrame

os.chdir("C:/Users/okigboo/Documents/Code/Code/Code/Section 2")


data1 = pd.read_csv("AAPL-Close-15-17.csv", index_col="date")
data1.head()
plt.plot(data1)
plt.show()

