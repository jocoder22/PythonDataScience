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
import pandas as pd

def pp2(*args):
    for arg in args:
        print(arg, end="\n\n")
                
def discounted_call_payoff(S,K, r, L, T):
    if (np.mean(S) > L):
        return 0
    return np.exp(-r*T)*np.maximum(S - K,0)
        
# calculate bond yield curve
num_months = 12
months = np.linspace(1,num_months,num_months)
par_value = 100
bondprices = np.array([99.38, 98.76,98.15,97.54,96.94,96.34,95.74,95.16,94.57,93.99,93.42,92.85])/100

yield_curve = -np.log(bondprices)/months

pp2(bondprices,yield_curve)

# plot bond yield and bond prices
plt.figure(figsize = [12, 8])
plt.plot(months, yield_curve)
plt.xlabel("Months")
plt.ylabel("yield")
plt.title('yield to Months')
# plt.legend()
plt.show()

# # Calibrate Libor forward rates
# # calibrating Vasicek model
# # Analytical bond price
np.random.seed(0)
r0 =  yield_curve[0]

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
  return sum(np.abs(bond_price_fun(r0, 0,months,alpha,b,sigma) - bondprices))

# # minimizing F function
bnds = ((0.00001,1),(0.00001,0.2), (0.00001,0.2))
opt_value = opt.fmin_slsqp(F, (0.03,0.05,0.03), bounds=bnds)
# opt_alpha = opt_value[0]
# opt_b = opt_value[1]
# opt_sigma = opt_value[2]
opt_alpha, opt_b, opt_sigma = opt_value

# Calculating model prices and yield
model_prices = bond_price_fun(r0,0,months, opt_alpha, opt_b, opt_sigma)
model_yield =  -np.log(model_prices)/months

pp2(opt_value)

df = pd.DataFrame({
      "MarketPrice":bondprices,
      "Modelprice": model_prices,
      "Marketyield": yield_curve,
      "Modelyield": model_yield
})

df["PriceDiff"] = df.MarketPrice - df.Modelprice
df["YieldDiff"] = df.Marketyield - df.Modelyield

pp2(df)

# plotting prices
plt.plot(t, bond_prices, label="Market prices")
plt.plot(t, model_prices, ".", label ="Calibarted prices")
plt.xlabel("Maturity")
plt.ylabel("Bond price")
plt.legend()
plt.show()

pp2(model_prices)

# For this submission, complete the following tasks: 
# 
#     1. Using a sample size of 100000, jointly simulate LIBOR forward rates, stock paths, 
#      and counterparty firm values.  
#     You should simulate the values monthly, and should have LIBOR forward rates applying over one 
#     month, starting one month apart, up to maturity. You may assume that the counterparty firm and stock 
#     values are uncorrelated with LIBOR forward  rates.

np.random.seed(0)
n_simulations = 100000
n_steps = 12
t =  np.linspace(1,num_months,num_months)
t_sim =  np.linspace(0,num_months - 1,num_months)


vasi_bond = bond_price_fun(r0,0,t_sim, opt_alpha, opt_b, opt_sigma)

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


for i in range(len(continuous_forwards)-1):
    print(np.exp(continuous_forwards[i+1])/np.exp(continuous_forwards[i])-1)

cont = []

cont.append(spot_rates[0])
for i in range(len(continuous_forwards)-1):
    cont.append(np.exp(continuous_forwards[i+1])/np.exp(continuous_forwards[i])-1)
    
# cont
annualized_rates = np.exp(np.array(cont)*12)-1
pp2(annualized_rates)

#     
#     2. Calculate the one-year discount factor which applies for each simulation, and use this to find first the value 
#     of the option for the jointly simulated stock and firm paths with no default risk, and then the value of the option 
#     with counterparty default risk. (Hint: you may want to use the reshape and ravel attributes of numpy arrays to ensure 
#     your dimensions match correctly. 

## Previous variabes from assignment 1
S0 = 100
T = 1
sigma = 0.3
K = 100
L = 150

correlation = 0.2
sigma_firm = 0.25
debt = 175
recovery_rate = 0.25
firm_value_0 = 200

## new variables
v0 = 0.06
kappa = 9
theta= 0.06
rho = -0.4
gamma = 0.75
k_log = np.log(K)

t_max = 30
N = 100

'''
Simulate a share price path using CEV local volatility terms 
and perform Monte Carlo simulations to calculate 
option prices with different sample sizes
'''
months_in_year = [x/12 for x in range(1,13)]
share_prices_dict = {}
cev_call_prices_dict = {}
mc_call_prices_dict = {}
std_cev_call_prices_dict = {}
std_mc_call_prices_dict = {}

share_prices = []

S_T = S0
prices = [S_T]
prices_mc = [S_T]
std = [0]

share_prices[0] = S_T
cev_call_prices = [np.maximum(S_T-K,0)]
mc_call_prices = [np.maximum(S_T-K,0)]
std_cev_call_prices = [0]
std_mc_call_prices = [0]

    
for t in months_in_year:
    vol = sigma*S_T**(gamma-1)
    delta_time = (1/12)
    Z_CEV = np.random.normal(0,1, n_simulations)

    ## Part 2: Share price path is calculated here with CEV local volatility terms
    S_T = S_T*np.exp((annualized_rates[int(t*12-1)]-(vol**2/2))*delta_time + (vol*np.sqrt(delta_time)*Z_CEV))
    share_prices.append(np.mean(S_T))

    
    if t == 1/12:
        print('Stock volatility drops as share price increases (vol vs S_T)')
    print(np.mean(vol), np.mean(S_T))

    ## Part 3. Perform Monte Carlo simulations
    mc_call_step = discounted_call_payoff(S_T, K, annualized_rates[int(t*12-1)], L, t)
    mc_call_prices.append(np.mean(mc_call_step))
    std_mc_call_prices.append(np.std(mc_call_step)/np.sqrt(n_simulations))# standard errors
   
    share_prices_dict[t] = share_prices
 
    mc_call_prices_dict[t] = mc_call_prices
    std_mc_call_prices_dict[t] = std_mc_call_prices

pp2(share_prices)

## Call price under CEV
z = 2 + 1/(1-gamma)
def CEV_call(S0,t,K):
    
    kappa2 = 2*annualized_rates[-1]/(sigma**2*(1-gamma)*(np.exp(2*annualized_rates[-1]*(1-gamma)*t)-1))
    x = kappa2*S0**(2*(1-gamma))*np.exp(2*annualized_rates[-1]*(1-gamma)*t)
    y = kappa2*K**(2*(1-gamma))
    return S0*(1-ncx2.cdf(y,z,x))-K*np.exp(-annualized_rates[-1]*t)*ncx2.cdf(x,z-2,y)

cev_call_price = CEV_call(S0,T,K)

print('Monte Carlo call price is: ', mc_call_prices[-1])
print('CEV call price is: ', cev_call_price)


# plotting prices
plt.figure(figsize = [10, 6])
plt.plot(np.arange(0,13),mc_call_prices, '.')
plt.plot(np.arange(0,13),mc_call_prices+3*np.array(std_mc_call_prices),'black')
plt.plot(np.arange(0,13),mc_call_prices-3*np.array(std_mc_call_prices),'g')
plt.xlabel("Months")
plt.ylabel("Price")
plt.title("Monte Carlo Estimates of risk-neutral call option price")
plt.legend(('Risk-neutral price', 'Risk-neutral price UB', 'Risk-neutral price LB'))
plt.show()



call_std_risky_dict = {}
cva_mean_risky_dict = {}
cva_std_risky_dict = {}
call_val_with_cva_risky_dict = {}
cev_call_prices_risky_dict = {}
mc_call_prices_risky_dict = {}
std_cev_call_prices_risky_dict = {}
std_mc_call_prices_risky_dict = {}

share_prices = []

S_T = S0
F_T = firm_value_0
prices = [S_T]
prices_mc = [S_T]

std = [0]
n_simulations = 100000

share_prices_risky = [S_T]
firm_prices_risky = [F_T]
cev_call_prices_risky = [np.maximum(S_T-K,0)]
mc_call_prices_risky = [np.maximum(S_T-K,0)]
std_cev_call_prices_risky = [0]
std_mc_call_prices_risky = [0]
corr_matrix = np.array([[1, correlation], [correlation,1]])
norm_matrix = norm.rvs(size = np.array([2,n_simulations]))
corr_norm_matrix = np.matmul(np.linalg.cholesky(corr_matrix), norm_matrix)


for t in months_in_year:
    vol = sigma*S_T**(gamma-1)
    delta_time = (1/12)
    
    ## Share price path is calculated here with simulated forward rates,
    ## CEV local volatility terms, and joint normal random variables
    S_T = S_T*np.exp((annualized_rates[int(t*12-1)]-(vol**2/2))*delta_time +(vol*np.sqrt(delta_time)*corr_norm_matrix[0,]))
    F_T = F_T*np.exp((annualized_rates[int(t*12-1)]-(vol**2/2))*delta_time +(vol*np.sqrt(delta_time)*corr_norm_matrix[1,]))
    share_prices_risky.append(np.mean(S_T))
    firm_prices_risky.append(np.mean(F_T))
    
    if t == 1/12:
        print("Monthly evolution of local volatility, share price and firm␣value")
    print(np.mean(vol), np.mean(S_T), np.mean(F_T))
    
    ## Perform Monte Carlo simulations
    call_val = discounted_call_payoff(S_T, K, annualized_rates[int(t*12-1)], L,t)
    amount_lost = np.exp(-annualized_rates[int(t*12-1)]*t)*(1-recovery_rate)*(F_T<debt)*call_val
    mc_call_step = discounted_call_payoff(S_T, K,annualized_rates[int(t*12-1)], L, t)
    
    mc_call_prices.append(np.mean(mc_call_step))
    std_mc_call_prices.append(np.std(mc_call_step)/np.sqrt(n_simulations)) # standard errors
    
    call_mean = np.mean(call_val)
    cva_mean = np.mean(amount_lost)
    
    call_std = np.std(call_val)/np.sqrt(n_simulations)
    cva_std = np.std(amount_lost)/np.sqrt(n_simulations)
    
    call_val_with_cva = call_mean - cva_mean
    
    call_mean_risky_dict[t] = call_mean
    call_std_risky_dict[t] = call_std
    cva_mean_risky_dict[t] = cva_mean
    cva_std_risky_dict[t] = cva_std
    
    call_val_with_cva_risky_dict[t] = call_val_with_cva
    

plt.figure(figsize=[12,8])
plt.plot([x*12 for x in months_in_year],list(call_mean_risky_dict.values()), '.')
plt.plot([x*12 for x in months_in_year],list(call_val_with_cva_risky_dict.values()),'-')
plt.plot([x*12 for x in months_in_year],list(call_mean_risky_dict.values())+3*np.array(list(call_std_risky_dict.values())),'black')
plt.plot([x*12 for x in months_in_year],list(call_mean_risky_dict.values())-3*np.array(list(call_std_risky_dict.values())),'g')
plt.xlabel("Months")
plt.ylabel("Price")
plt.title("Monte Carlo Estimates of risk-adjusted call option price")
plt.legend(('Risk-neutral price', 'Risk-adjusted price', 'Risk-neutral price␣UB', 'Risk-neutral price LB'))
plt.show()


print("Risk-neutral call price at the end of 12 months is: ", list(call_mean_risky_dict.values())[-1])
print("Risk-adjusted call price at the end of 12 months is: ", list(call_val_with_cva_risky_dict.values())[-1])

