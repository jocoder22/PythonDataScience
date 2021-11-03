#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import  yfinance as yf

symbols = "KO XOM T VOD JPM C NVDA".split()

assets =  yf.download(tickers = symbols[0], start="2020-01-01", auto_adjust = True, progress=False)


def format_borders(plot, title, ylabel):
    """
    
    
    
    """
    config_ticks = {'size': 14, 'color': 'grey', 'labelcolor': 'grey'}
    config_title = {'size': 18, 'color': 'grey', 'ha': 'left', 'va': 'baseline'}        
    plot.spines['top'].set_visible(False)
    plot.spines['left'].set_visible(False)
    plot.spines['left'].set_color('grey')
    plot.spines['bottom'].set_color('grey')

    plot.yaxis.tick_right()
    plot.tick_params(axis='both', **config_ticks)
    plot.set_ylabel(ylabel, fontsize=14)
    plot.yaxis.set_label_position("right")
    plot.yaxis.label.set_color('grey')
    plot.grid(axis='y', color='gainsboro', linestyle='-', linewidth=0.5)
    plot.set_axisbelow(True)
    plot.set_title(title, **config_title)



fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                gridspec_kw={'height_ratios': [3, 2]}, tight_layout = {"pad": 1})

axes[0].plot(assets.index, assets["Close"], color='blue', linewidth=2, label='Price')
axes[1].bar(assets.index, assets["Volume"], color='darkgrey', width=3, label='Volume')
format_borders(axes[0], "Cocakola Daily close Prices", 'Price (in USD)')
format_borders(axes[1], "Cocakola Daily Volumes", "Volumes (in Millions)")
plt.show()

