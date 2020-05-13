#!/usr/bin/env python
# coding: utf-8

# # Submission 3: 
# ## Simulate Asset Price Evolutions and Reprice Risky up-and-out Call Option  
# The goal of Submission 3 is to reprice the risky up-and-out call option from Submission 1, but now implementing 
# a non-constant interest rate<br>  and local volatility. With the exception of the interest rates and volatilities, 
# you may make the same assumptions as in Submission 1: 
# - Option maturity is one year  
# - The option is struck at-the-money 
# - The up-and-out barrier for the option is \\$150 
# - The current share price is \\$100 
# - The current firm value for the counterparty is \\$200 
# - The counterparty’s debt, due in one year, is \\$175 
# - The correlation between the counterparty and the stock is constant at 0.2 
# - The recovery rate with the counterparty is 25%. 
# 
# 
# The local volatility functions for both the stock and the counterparty have the same form as in part 2, namely $\sigma(t_i,t_{i+1}) = \sigma(S_{ti})^{\gamma -1}$. For the stock<br>  $\sigma_S(t_i,t_{i+1}) = 0.3\sigma(S_{ti})^{\gamma -1}$, and for the counterparty, $\sigma_V(t_i,t_{i+1}) = \sigma(V_{ti})^{\gamma -1}$, where $\gamma = 0.75$. We can simulate the next step in a share price<br> path using the following formula: 
# 
# $$S_{t_{i+1}} = S_{t_i}e^{\left(r \,\, -  \,\frac{\sigma^2\left(t_i,\,t_{i+1}\right)}{2}\right)}(t_{i+1}\, - \,t_i) + \sigma(t_i,t_{i+1})\sqrt{t_{i+1}\, - \,t_i}Z $$
# 
# 
# where $𝑆_{ti}$ is the share price at time $𝑡_i$, $\sigma(t_i,t_{i+1})$  is the volatility for the period $[𝑡_i,𝑡_{𝑖+1}]$, $𝑟_{𝑡𝑖}$ is the risk-free interest rate, and 𝑍~𝑁(0,1). The<br>counterparty firm values can be simulated similarly. You observe the following zero-coupon bond prices (per $100 nominal) in the market: 
# 
# |Maturity|Price|
# |------|------|
# |1 Month  |$99.38  |
# |2 Months  |\$98.76 |
# |3 Months  |\$98.15  |
# |4 Months  |\$97.54 |
# |5 Months |\$96.94  |
# |6 Months |\$96.34 |
# |7 Months  |\$95.74   |
# |8 Months  |\$95.16 |
# |9 Months  |\$94.57 |
# |10 Months  |\$93.99  |
# |11 Months  |\$93.42  |
# |12 Months  |\$92.85 |
# 
# 

# You are required to use a LIBOR forward rate model to simulate interest rates. The initial values for the LIBOR forward rates need to be calibrated to the market forward rates which can be deduced through the market zero-coupon bond prices given above. This continuously compounded interest rate for $[𝑡_i,𝑡_{𝑖+1}]$ at time $t_i$, is then given by the solution to:
# 
# $$e^{r_{ti}\left(t_{i+1}-t_i\right)} = 1 + L(𝑡_i,𝑡_{𝑖+1})(t_{i+1}\, - \,t_i)$$
# 
# Where $L(𝑡_i,𝑡_{𝑖+1})$ is the LIBOR forward rate which applies from $t_i$ to $t_{i+1}$, at time $t_i$. Note that these LIBOR rates are updated as you run through the simulation, and so your continuously compounded rates should be as well. 


# import required modules
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.stats import ncx2
import scipy.optimize as opt
import math
import random

def pp2(*args):
    for arg in args:
        print(arg, end="\n\n")
        
# calculate bond yield curve
num_months = 12
# months = np.arange(1,num_months+1)
months = np.linspace(1,12,12)
nomialbondprice = 100
bondprices = np.array([99.38, 98.76,98.15,97.54,96.94,96.34,95.74,95.16,94.57,93.99,93.42,92.85])

spot_rates = -np.log(bondprices/nomialbondprice)/months *100
nominal_est = np.exp(spot_rates * months / 100)* bondprices

pp2(spot_rates, spot_rates[0])

# plot bond yield and bond prices
plt.figure(figsize = [12, 8])
plt.plot(months, spot_rates)
plt.xlabel("Months")
plt.ylabel("Spot rate")
plt.title('Spot rate to Months')
# plt.legend()
plt.show()

# For this submission, complete the following tasks: 
# 
#     1. Using a sample size of 100000, jointly simulate LIBOR forward rates, stock paths, and counterparty firm values.  
#     You should simulate the values monthly, and should have LIBOR forward rates applying over one month, starting one month apart, up to maturity. You may assume that the counterparty firm and stock values are uncorrelated with LIBOR forward  
#     rates.
#     
#     2. Calculate the one-year discount factor which applies for each simulation, and use this to find first the value 
#     of the option for the jointly simulated stock and firm paths with no default risk, and then the value of the option 
#     with counterparty default risk. (Hint: you may want to use the reshape and ravel attributes of numpy arrays to ensure 
#     your dimensions match correctly. 


# # Calibrate Libor forward rates
# # calibrating Vasicek model
# # Analytical bond price
np.random.seed(0)
r0 =  spot_rates[0]/100
t = np.linspace(1,12,12)

def A(t1,t2,alpha):
    return (1-np.exp(-alpha*(t2-t1)))/alpha

def D(t1, t2, alpha, b , sigma):
  val1 = (t2-t1-A(t1, t2, alpha))*(sigma**2/(2*alpha**2)-b)
  val2 = sigma**2*A(t1,t2,alpha)**2/(4*alpha)
  return val1 - val2

def bond_price_fun(r,t,T,alpha,b,sigma):
    return np.exp(-A(t,T,alpha)*r+D(t,T,alpha,b,sigma))

# the F function
# find the values for which the differences between the bondprice and yield curve bond prices are minimized
def F(x):
  alpha = x[0]
  b = x[1]
  sigma = x[2]
  return sum(np.abs(bond_price_fun(r0, 0,t,alpha,b,sigma) - bondprices))

# # minimizing F function
bnds = ((0.00001,1),(0.00001,0.2), (0.00001,0.2))
opt_value = opt.fmin_slsqp(F, (0.03,0.05,0.03), bounds=bnds)
opt_alpha = opt_value[0]
opt_b = opt_value[1]
opt_sigma = opt_value[2]

# Calculating model prices and yield
model_prices = bond_price_fun(r0,0,t, opt_alpha, opt_b, opt_sigma)
model_yield =  -np.log(model_prices/nomialbondprice)/t

pp2(opt_value)

# plotting prices
plt.plot(t, bond_prices, label="Market prices")
plt.plot(t, model_prices, ".", label ="Calibarted prices")
plt.xlabel("Maturity")
plt.ylabel("Bond price")
plt.legend()
plt.show()

pp2(model_prices)

# ###### Plan of Attack
# 1. Find the monthly forward rates implied by the ZCB yield curve.
# 2. Convert the monthly forward rates to continuously compounded rates.
# 3. Calibrate the interest rates given by Vasicek model using the continuously compounded rates.
# 4. Find the CEV local volatility term for each month.
# 5. Using CEV local volatility terms and calibrated interest rates, estimate monthly stock price path and counterparty firm values.
# 6. Calculate the one-year discount factor
# 7. Find the option price assuming no counterparty default risk
# 8. Find the option price assuming counterparty default risk

np.random.seed(0)
n_simulations = 100000
# 12 months
n_steps = 12
t =  np.linspace(1,12,12)

pp2(spot_rates[0]/100)

alpha = opt_val[0]
b = opt_val[1]
sigma = opt_val[2]
r0 =  spot_rates[0]/100

vasi_bond = bond_price_fun(r0,0,t, alpha, b, sigma)
pp2((alpha, b, sigma), vasi_bond)

analytic_bondprices = model_prices

# mc_forward = np.ones([n_simulations, n_steps])*(bond_prices[:])/(2*bond_prices[:])
# mc_forward = np.ones([n_simulations, n_steps-1])*(analytic_bondprices[:-1]-analytic_bondprices[1:])/(1*analytic_bondprices[1:])
# predcorr_forward = np.ones([n_simulations, n_steps-1])*(analytic_bondprices[:-1]-analytic_bondprices[1:])/(1*analytic_bondprices[1:])

mc_forward = np.ones([n_simulations, n_steps-1])*(vasi_bond[:-1]-vasi_bond[1:])/(1*vasi_bond[1:])
predcorr_forward = np.ones([n_simulations, n_steps-1])*(vasi_bond[:-1]-vasi_bond[1:])/(1*vasi_bond[1:])


predcorr_capfac = np.ones([n_simulations,n_steps])
mc_capfac = np.ones([n_simulations,n_steps])
delta = np.ones([n_simulations,n_steps-1])*(t[1:]-t[:-1])
sigmaj =.2
mean_muhat = []
mean_mcforward = np.ones([n_steps, n_steps])


for i in range(1,n_steps):
    Z = norm.rvs(size = [n_simulations,1])
    
    muhat = np.cumsum(delta[:,i:]*mc_forward[:,i:]*sigmaj**2/(1+delta[:,i:]*mc_forward[:,i:]),axis =1)
    mc_forward[:,i:] = mc_forward[:,i:]*np.exp((muhat-sigmaj**2/2)*delta[:,i:]+sigmaj*np.sqrt(delta[:,i:])*Z)
    
    # Predictor-Corrector Montecarlo simulation
    mu_initial = np.cumsum(delta[:,i:]*predcorr_forward[:,i:]*sigmaj**2/(1+delta[:,i:]*predcorr_forward[:,i:]),axis = 1)
    for_temp = predcorr_forward[:,i:]*np.exp((mu_initial-sigmaj**2/2)*delta[:,i:]+sigmaj*np.sqrt(delta[:,i:])*Z)
    mu_term = np.cumsum(delta[:,i:]*for_temp*sigmaj**2/(1+delta[:,i:]*for_temp),axis = 1)
    predcorr_forward[:,i:] = predcorr_forward[:,i:]*np.exp((mu_initial+mu_term-sigmaj**2)*delta[:,i:]/2+sigmaj*np.sqrt(delta[:,i:])*Z)


# Implying capitalisation factors from the forward rates
mc_capfac[:,1:] = np.cumprod(1+delta*mc_forward, axis = 1)
predcorr_capfac[:,1:] = np.cumprod(1+delta*predcorr_forward, axis = 1)

# Inverting the capitalisation factors to imply bond prices (discount factors)
mc_price = mc_capfac**(-1)
predcorr_price = predcorr_capfac**(-1)

# Taking averages
mc_final = np.mean(mc_price,axis = 0)
predcorr_final = np.mean(predcorr_price,axis = 0)


# plotting prices
plt.figure(figsize = [10, 6])
plt.plot(t, vasi_bond, label="Vesicek bond prices")
plt.plot(t, mc_final, label ="Mento carlos bond prices")
plt.plot(t, predcorr_final, "x", label ="Predictor corrected prices")
plt.xlabel("Maturity")
plt.ylabel("Bond price")
plt.legend()
plt.show()


pp2(vasi_bond - mc_final,predcorr_final )

model_yield = 1/predcorr_final - 1

continuous_forwards = np.log(1 + model_yield)
pp2(continuous_forwards)

np.exp(continuous_forwards[-1])/np.exp(continuous_forwards[-2])


for i in range(len(continuous_forwards)-1):
    print(np.exp(continuous_forwards[i+1])/np.exp(continuous_forwards[i])-1)

cont = []

cont.append(spot_rates[0]/100)
# cont = cont + [x for x in continuous_forwards]
for i in range(len(continuous_forwards)-1):
    cont.append(np.exp(continuous_forwards[i+1])/np.exp(continuous_forwards[i])-1)
    
# cont
cont_monthly = np.exp(np.array(cont)*12)-1


## Previous variabes from assignment 1
S0 = 100
T = 1
sigma = 0.3
K = 100
# r = 0.08

## new variables
v0 = 0.06
kappa = 9
theta= 0.06
rho = -0.4
gamma = 0.75
k_log = np.log(K)

t_max = 30
N = 100


###CEV local volatility term is incorporated into S_T
def discounted_call_payoff(S_T,K,risk_free_rate,time):
    return np.exp(-risk_free_rate*time)*np.maximum(S_T-K,0)

'''
Simulate a share price path using CEV local volatility terms 
and perform Monte Carlo simulations to calculate 
option prices with different sample sizes
'''
months_in_year = [x/12 for x in range(1,13)]
# share_prices_dict = {}
# cev_call_prices_dict = {}
# mc_call_prices_dict = {}
# std_cev_call_prices_dict = {}
# std_mc_call_prices_dict = {}

share_prices = []

# share_prices_dict[0] = S0
#norm_array = norm.rvs(size = 1000*i)
# S_T = S0
# prices = [S_T]
# prices_mc = [S_T]
# std = [0]

# share_prices[0] = S_T
# cev_call_prices = [np.maximum(S_T-K,0)]
# mc_call_prices = [np.maximum(S_T-K,0)]
# std_cev_call_prices = [0]
# std_mc_call_prices = [0]

for i in range(1,n_steps):
    ## Added for monte carlo
    #norm_array = norm.rvs(size = 1000*i)
#     S_T = S0
#     prices = [S_T]
#     prices_mc = [S_T]
#     std = [0]
    
#     share_prices = [S_T]
#     cev_call_prices = [np.maximum(S_T-K,0)]
#     mc_call_prices = [np.maximum(S_T-K,0)]
#     std_cev_call_prices = [0]
#     std_mc_call_prices = [0]
    
#     for indx, t in enumerate(months_in_year):
    r = cont_monthly[i]
    vol = sigma*S_T**(gamma-1)
    delta_time = (1/12)

#         Z = norm.rvs(size = [n_simulations,1])
    Z_CEV = norm.rvs(size = n_simulations)
#         Z_CEV = np.random.normal(0,1, i*1000)

    ## Part 2: Share price path is calculated here with CEV local volatility terms
    S_T = S_T*np.exp((r-(vol**2/2))*delta_time + (vol*np.sqrt(delta_time)*Z_CEV))
    share_prices.append(np.mean(S_T))

    if i == 50:
        if t == 1/12:
            print('Stock volatility drops as share price increases (vol vs S_T)')
        print(np.mean(vol), np.mean(S_T))

    ## Part 3. Perform Monte Carlo simulations
    mc_call_step = discounted_call_payoff(S_T, K, r, t)
    mc_call_prices.append(np.mean(mc_call_step))
    std_mc_call_prices.append(np.std(mc_call_step)/np.sqrt(n_simulations))# standard errors
   
#     share_prices.append(share_prices)
 
    mc_call_prices_dict[i] = mc_call_prices
    std_mc_call_prices_dict[i] = std_mc_call_prices

pp2(share_prices)

## Call price under CEV
z = 2 + 1/(1-gamma)
def CEV_call(S0,t,K):
    
    kappa2 = 2*r/(sigma**2*(1-gamma)*(np.exp(2*r*(1-gamma)*t)-1))
    x = kappa2*S0**(2*(1-gamma))*np.exp(2*r*(1-gamma)*t)
    y = kappa2*K**(2*(1-gamma))
    return S0*(1-ncx2.cdf(y,z,x))-K*np.exp(-r*t)*ncx2.cdf(x,z-2,y)

cev_call_price = CEV_call(S0,T,K)

print('Monte Carlo call price is (N=50,000 and T = 1): ', mc_call_prices[-1])
print('CEV call price is: ', cev_call_price)
