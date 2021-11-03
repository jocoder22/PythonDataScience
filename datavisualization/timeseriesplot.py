#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import  yfinance as yf

symbols = "KO XOM T VOD JPM C NVDA".split()

assets =  yf.download(tickers = symbols[0], start="2020-01-01", auto_adjust = True, progress=False)

print(assets.head())