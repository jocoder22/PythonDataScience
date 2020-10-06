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

def prod(mu, sigma, samples):
    return reduce(operator.mul, 
                  list(map(lambda x: plot(mu,sigma, x), 
                           range(1,samples+1))))