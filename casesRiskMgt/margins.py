#!/usr/bin/env python
import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functools import reduce
import operator

import holoviews as hv
import hvplot.pandas

from printdescribe import print2

np.random.seed(42)
hv.extension('bokeh')


def plot(mu, sigma, samples):
    return pd.Series(np.random.normal(mu,sigma, 1000)).cumsum(
    ).hvplot(title='Random Walks', label=f'{samples}')


# generate values and plot
def prod(mu, sigma, samples):
    return reduce(operator.mul, 
                  list(map(lambda x: plot(mu,sigma, x), 
                           range(1,samples+1))))


# draw dynamic graphs
hv.DynamicMap(prod,kdims=['mu', 'sigma','samples']).redim.range(
    mu=(0,5), 
    sigma=(1,10), 
    samples=(2,10)
    ).options(width=900, height=400) 


def simulation_prices(days=100, runs=1000, axis=0):
    run = []
    
    for _ in range(runs):
        a = Accounts()
        prices = pd.DataFrame([a.price()[0] for day in range(days)], 
                              columns=['return'])
        run.append(prices)
        
        
    output = pd.concat(run, axis=axis)
    output.columns = [f'Run {i+1}' for i in range(output.shape[1])]
    
    return output