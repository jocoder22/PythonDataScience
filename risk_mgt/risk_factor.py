import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import pandas_datareader as pdr
from pandas.util.testing import assert_frame_equal


def print2(*args):
    for arg in args:
        print(arg, end="\n\n")

stocklist = ["C","JPM","MS", "GS"]
p_labels = ["Citibank", "J.P. Morgan", "Morgan Stanley", "Goldman Sachs"]

starttime = datetime.datetime(2000, 1, 1)

# get only the closing prices
portfolio = pdr.get_data_yahoo(stocklist, starttime)['Close']

# set the weights
weights = [0.20, 0.30, 0.30, 0.20]

# calculate percentage return and portfolio return
asset_returns = portfolio.pct_change()
returns = asset_returns.dot(weights)

# path = r"D:\PythonDataScience\risk_mgt\DRSFRMACBS.csv"
weblink = ("https://fred.stlouisfed.org/graph/fredgraph.csv?bgcolor=%23e1e9f0&chart_type="
        "line&drp=0&fo=open%20sans&graph_bgcolor=%23ffffff&height=450&mode=fred&recession_bars="
        "on&txtcolor=%23444444&ts=12&tts=12&width=1168&nt=0&thu=0&trc=0&show_legend="
        "yes&show_axis_titles=yes&show_tooltip=yes&id=DRSFRMACBS&scale=left&cosd=2000-01-01&coed="
        "2019-10-01&line_color=%234572a7&link_values=false&line_style=solid&mark_type=none&mw="
        "3&lw=2&ost=-99999&oet=99999&mma=0&fml=a&fq=Quarterly%2C%20End%20of%20Period&fam="
        "avg&fgst=lin&fgsnd=2009-06-01&line_index=1&transformation=lin&vintage_date="
        "2020-05-29&revision_date=2020-05-29&nd=1991-01-01")

# df = pd.read_csv(weblink, parse_dates=True, index_col=0, names=["Mortage Deliquency Rate"])
df = pd.read_csv(weblink).set_index("DATE")
df.columns = ["Mortage Deliquency Rate"]
print2(df.loc["2005-03-31":], returns)
