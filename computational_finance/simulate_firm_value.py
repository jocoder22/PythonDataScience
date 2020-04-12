#!/usr/bin/env python
import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

def print2(*args):
    for arg in args:
        print(arg, sep="\n\n", end="\n\n")

np.random.seed(100)
V0 = 1000	#the initial firm value
mu = 0.15	#miu parameter in the Geometric Brownian Motion (GBM) 
sigma = 0.13	#volatility parameter in GBM
dt = 1 / 252	#the length of delta(t)
T = 1		#the end of time period	
M1 = T/dt	#the number of time periods (in our case the number of days in a year)
n1 = 10000	#the number of generated trajectories 


mean = (mu - sigma**2 / 2) * dt
sd = sigma * dt**0.5

val = np.random.normal(mean, sd, int(n1*M1))

val2 = val.reshape(int(M1), n1)

df = pd.DataFrame(val2)
df3 = df.cumsum(axis=0)
d5 = V0 * df3.apply(lambda x: np.exp(x))
x = np.arange(0, T, dt)

plt.figure(figsize = [11, 6])
plt.plot(x, d5[[1, 2,3, 4,5]])
plt.xlabel('Time',)
plt.ylabel('Firm value trajectories')
plt.title('Trajectories of firm values in the Merton model')
plt.show()


S = 1000000
T = 252
mu = 1.92246
vol = .86123  #Between .76 and .86

result = []
for i in range(10):
    daily_returns = np.random.normal(mu/T,vol/np.sqrt(T),T)+1
    
    price_list = [S]
    
    for x in daily_returns:
        price_list.append(price_list[-1]*x)
    result.append(price_list[-1]) #appending each runs end value --to calculate the mean return
        
    plt.plot(price_list)#This is key, KEEP THIS IN LOOP,votherwise it will plot one iteration/return path.
plt.title('Daily MC')
plt.show()



# # Create portfolio returns column
# returns['Portfolio']= returns.dot(weights)

# # Calculate cumulative returns
# daily_cum_ret=(1+returns).cumprod()

# # Plot the portfolio cumulative returns only
# fig, ax = plt.subplots()
# ax.plot(daily_cum_ret.index, daily_cum_ret.Portfolio, color='purple', label="portfolio")
# ax.xaxis.set_major_locator(matplotlib.dates.YearLocator())
# plt.legend()
# plt.show()

