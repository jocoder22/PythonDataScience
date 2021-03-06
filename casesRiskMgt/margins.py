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



%%opts Curve [width=500 height=300]
hv.DynamicMap(simulation_plots, kdims=['days', 'runs']).redim.range(days=(100,500), runs=(5,15)).options(width=900, height=400)


class Accounts:
    def __init__(self, account_mu=20, account_sigma=5, account_numbers=100, mu=0, sigma=0.05, margin_mu=0.1, momentum_mu=0.025, margin_sigma=0.001):
        # We initialize paramters for our simulations
        self.account_mu=account_mu
        self.account_sigma = account_sigma
        self.account_numbers = account_numbers
        
        self.margin_mu=margin_mu
        self.momentum_mu=momentum_mu
        self.margin_sigma=margin_sigma
        
        self.accounts= np.maximum(np.random.normal(loc=self.account_mu, scale=self.account_sigma, size = self.account_numbers),0)
        self.call=self.accounts*np.random.uniform(0.5,0.7,self.account_numbers)
               
        self.mu=mu
        self.sigma=sigma
        
        self.called_accounts_factor = 0
        
        self.momentum = 0
        self.history = [0,0,0,0,0]
        
        return None
    
    def price(self):
        # Calculate factors
        self.called_accounts_factor =((self.accounts <= self.call).sum())/self.account_numbers
        self.momentum = (self.history[4] - self.history.pop(0))
        
        # Update paramteres
        self.mu = self.mu - self.margin_mu*self.called_accounts_factor + self.momentum_mu*self.momentum
        self.sigma = self.sigma + self.margin_sigma*self.called_accounts_factor
        
        # Update accounts
        self.history.append(np.random.normal(loc=self.mu, scale=self.sigma))
        self.accounts = self.accounts*(1+self.history[4]*np.random.uniform(low=0.5,high=1.5,size=self.account_numbers))
        
        # Reset called accounts
        reset = (np.random.rand(self.account_numbers)>=0.3) * (self.accounts <= self.call)
        self.accounts[reset] = np.random.normal(self.account_mu, self.account_sigma)
        self.call[reset] = self.accounts[reset]*np.random.uniform(0.2,0.5,np.sum(reset))

        return [self.history[4], self.accounts.sum(), self.called_accounts_factor, self.momentum]

# define funtion to simulate prices
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

%%opts Curve [width=500 height=300]
hv.DynamicMap(simulation_plots, kdims=['days', 'runs']).redim.range(days=(100,500), runs=(5,15)).options(width=900, height=400)

# initiate the simulation
simulations = simulation_prices()


%%opts Overlay [show_title=True] Distribution [height=500, width=1000]
hv.Distribution(np.random.normal(simulations.mean(),simulations.std(),100000), label='Normal') * hv.Distribution(simulations.iloc[:,0], label='Simulation').options(fill_alpha=0.0)
