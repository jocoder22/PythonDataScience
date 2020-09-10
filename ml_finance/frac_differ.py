import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pkgutil
import mlfinlab

from mlfinlab.features.fracdiff import FractionalDifferentiation as ff
from mlfinlab.features.fracdiff import FractionalDifferentiation
from mlfinlab.features.fracdiff import frac_diff, plot_min_ffd, frac_diff_ffd

from statsmodels.tsa.stattools import adfuller
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



package=mlfinlab
for importer, modname, ispkg in pkgutil.walk_packages(path=package.__path__,
                                                      prefix=package.__name__+'.',
                                                      onerror=lambda x: None):
    print(modname)

    

fracdata = ff.frac_diff(stock, 0.2, thresh=1e3)
print(fracdata.head())

fracdata.columns = ["frac0.2"]

result = pd.concat([stock,fracdata], axis=1, sort=False)
result["logprice"] = np.log(result['Close'])
result["logdiff"] = np.log(result['Close']/result['Close'].shift(1))
result["frac0.5"] = ff.frac_diff(stock, 0.5, thresh=1e3)
result.dropna(inplace=True)
print(result.head())


sns.relplot(height=6, data=result, kind="line")
plt.show();


def plot_min_ffd2(series):
    """
    Advances in Financial Machine Learning, Chapter 5, section 5.6, page 85.

    References:

    * https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086

    This function plots the graph to find the minimum D value that passes the ADF test.

    It allows to determine d - the amount of memory that needs to be removed to achieve
    stationarity. This function covers the case of 0 < d << 1, when the original series is
    "mildly non-stationary."

    The right y-axis on the plot is the ADF statistic computed on the input series downsampled
    to a daily frequency.

    The x-axis displays the d value used to generate the series on which the ADF statistic is computed.

    The left y-axis plots the correlation between the original series (d=0) and the differentiated
    series at various d values.

    Examples on how to interpret the results of this function are available in the corresponding part
    in the book Advances in Financial Machine Learning.

    :param series: (pd.DataFrame) Dataframe that contains a 'close' column with prices to use.
    :return: (plt.AxesSubplot) A plot that can be displayed or used to obtain resulting data.
    """

    results = pd.DataFrame(columns=['adfStat', 'pVal', 'lags', 'nObs', '95% conf', 'corr'])

    # Iterate through d values with 0.1 step
    for d_value in np.linspace(0, 1, 11):
        # close_prices = np.log(series[['close']]).resample('1D').last()  # Downcast to daily obs
        close_prices = np.log(series[['close']])
        close_prices.dropna(inplace=True)
        print2(" ")
        print2(close_prices)

        # Applying fractional differentiation
        differenced_series = frac_diff_ffd(close_prices, diff_amt=d_value, thresh=0.01).dropna()

        # Correlation between the original and the differentiated series
        corr = np.corrcoef(close_prices.loc[differenced_series.index, 'close'],
                           differenced_series['close'])[0, 1]
        # Applying ADF
        differenced_series = adfuller(differenced_series['close'], maxlag=1, regression='c', autolag=None)

        # Results to dataframe
        results.loc[d_value] = list(differenced_series[:4]) + [differenced_series[4]['5%']] + [corr]  # With critical value

    # Plotting
    plot = results[['adfStat', 'corr']].plot(secondary_y='adfStat', figsize=(10, 8))
    plt.axhline(results['95% conf'].mean(), linewidth=1, color='r', linestyle='dotted')

    return plot

stock.columns = ["close"]
ax = plot_min_ffd2(stock)
plt.show()
