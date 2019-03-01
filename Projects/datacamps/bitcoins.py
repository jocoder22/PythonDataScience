#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# plt.style.use('ggplot')

url = 'https://api.coinmarketcap.com/v1/ticker/?limit=0%20named%20datasets/coinmarketcap_06122017.csv'
js = pd.read_json(url)
data = pd.DataFrame(js)