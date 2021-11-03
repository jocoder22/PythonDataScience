#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import  yfinance as yf

symbols = "KO XOM T VOD JPM C NVDA".split()

assets =  yf.download(tickers = symbols[0], start="2020-01-01", auto_adjust = True, progress=False)

print(assets.head(), assets.index)

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                gridspec_kw={'height_ratios': [3, 1]}, tight_layout = {"pad":3})
# fig.tight_layout(pad=3)

axes[0].plot(assets.index, assets["Close"], color='blue', linewidth=2, label='Price')
axes[1].bar(assets.index, assets["Volume"], color='darkgrey', width=3, label='Volume')
plt.show()

