import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from printdescribe import print2

sns.relplot(data=pd.Series(np.random.normal(0,1,1000)))
plt.show();

np.random.seed(4)
sns.relplot(data=pd.Series(
    np.cumsum(np.random.normal(0,1,1000))))
plt.show();
