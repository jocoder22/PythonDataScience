import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import pandas_datareader as pdr
from datetime import datetime, date
from printdescribe import print2

sns.relplot(data=pd.Series(np.random.normal(0,1,1000)))
plt.show();

np.random.seed(4)
sns.relplot(data=pd.Series(
    np.cumsum(np.random.normal(0,1,1000))))
plt.show();


start = datetime(2010, 6, 29)
end = datetime(2018, 3, 27)
symbol = 'GOOG'

stock = pdr.get_data_yahoo(symbol, start, end)[['Close']]
print2(stock.head())


import pkgutil
import mlfinlab
package=mlfinlab
for importer, modname, ispkg in pkgutil.walk_packages(path=package.__path__,
                                                      prefix=package.__name__+'.',
                                                      onerror=lambda x: None):
    print(modname)

    
from mlfinlab.features.fracdiff import FractionalDifferentiation as ff
fracdata = ff.frac_diff(stock, 0.2, thresh=1e3)
print(fracdata.head())

fracdata.columns = ["frac0.2"]

result = pd.concat([stock,fracdata], axis=1, sort=False)
result["logprice"] = np.log(result['Close'])
result["logdiff"] = np.log(result['Close']/result['Close'].shift(1))
result["frac0.5"] = ff.frac_diff(stock, 0.5, thresh=1e3)
result.dropna(inplace=True)
print(result.head())


sns.relplot(height=8, data=result, kind="line")
plt.show();
