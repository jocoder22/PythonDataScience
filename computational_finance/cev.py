#Importing libraries
import numpy as np
from scipy.stats import norm
from scipy.stats import uniform
import matplotlib.pyplot as plt
from scipy.stats import ncx2
import math
import random

sp = {"end": "\n\n", "sep":"\n\n"}
np.random.seed(100)

#Share specific information
S0 = 100
v0 = 0.06
kappa = 9
theta = 0.06
r = risk_free = 0.03
sigma = 0.5
rho = -0.4
sigma = 0.5

gamma = 0.75

#Call Option specific information
K = 100
T = 0.5
k_log = np.log(K)

#Approximation information
t_max = 30
N = 100




S0 = 100
v0 = 0.06
kappa = 9
theta = 0.06
r = 0.08
sigma = 0.3
rho = -0.4

#Variable declaration
# S0 = 100
sigma = 0.3
gamma = 0.75
# r = 0.1
# T = 3



#Call Option specific information
K = 100
T = 1
k_log = np.log(K)

#Approximation information
t_max = 30
N = 100
# ## new variables
# S0 =100
# v0 = .06
# kappa = 9
# theta= .06
# r = .08
# sigma = .3
# rho = -.4
# gamma = .75
# T=.5
# k_log = np.log(K)

# t_max = 30
# N = 100


z = 2 + 1/(1-gamma)
def C(t,K):
    kappa = 2*r/(sigma**2*(1-gamma)*(np.exp(2*r*(1-gamma)*t)-1))
    x = kappa*S0**(2*(1-gamma))*np.exp(2*r*(1-gamma)*t)
    y = kappa*K**(2*(1-gamma))
    return S0*(1-ncx2.cdf(y,z,x))-K*np.exp(-r*t)*ncx2.cdf(x,z-2,y)
    
test_strikes = np.linspace(80,120,41)

delta_t = 0.01
delta_K = 0.01
dC_dT = (C(T+delta_t,test_strikes)-C(T-delta_t,test_strikes))/(2*delta_t)
dC_dK = (C(T,test_strikes+delta_K)-C(T,test_strikes-delta_K))/(2*delta_K)
d2C_dK2 = (C(T,test_strikes+2*delta_K)-2*C(T,test_strikes+delta_K)+C(T,test_strikes))/(delta_K**2)

vol_est = np.sqrt(2)/test_strikes*np.sqrt((dC_dT+r*test_strikes*dC_dK)/d2C_dK2)

delta_t = t_max/N
from_1_to_N = np.linspace(1,N,N)
t_n = (from_1_to_N-1/2)*delta_t


# Code for analytical solution for vanilla European Call option
d_1_stock = (np.log(S0/K)+(r + sigma**2/2)*(T))/(sigma*np.sqrt(T))
d_2_stock = d_1_stock - sigma*np.sqrt(T)

analytic_callprice = S0*norm.cdf(d_1_stock)-(K*np.exp(-r*(T))*norm.cdf(d_2_stock))
print(analytic_callprice)

#Characteristic function code

a = sigma**2/2

def b(u):
    return kappa - rho*sigma*1j*u


def c(u):
    return -(u**2+1j*u)/2


def d(u):
    return np.sqrt(b(u)**2-4*a*c(u))

def xminus(u):
    return (b(u)-d(u))/(2*a)


def xplus(u):
    return (b(u)+d(u))/(2*a)


def g(u):
    return xminus(u)/xplus(u)


def C(u):
    val1 = T*xminus(u)-np.log((1-g(u)*np.exp(-T*d(u)))/(1-g(u)))/a
    return r*T*1j*u + theta*kappa*val1


def D(u):
    val1 = 1-np.exp(-T*d(u))
    val2 = 1-g(u)*np.exp(-T*d(u))
    return (val1/val2)*xminus(u)


def log_char(u):
    return np.exp(C(u) + D(u)*v0 + 1j*u*np.log(S0))

def adj_char(u):
    return log_char(u-1j)/log_char(-1j)


def g(u):
    return xminus(u)/xplus(u)


def C(u):
    val1 = T*xminus(u)-np.log((1-g(u)*np.exp(-T*d(u)))/(1-g(u)))/a
    return r*T*1j*u + theta*kappa*val1


def D(u):
    val1 = 1-np.exp(-T*d(u))
    val2 = 1-g(u)*np.exp(-T*d(u))
    return (val1/val2)*xminus(u)


def log_char(u):
    return np.exp(C(u) + D(u)*v0 + 1j*u*np.log(S0))


def adj_char(u):
    return log_char(u-1j)/log_char(-1j)


first_integral = sum((((np.exp(-1j*t_n*k_log)*adj_char(t_n)).imag)/t_n)*delta_t)
second_integral = sum((((np.exp(-1j*t_n*k_log)*log_char(t_n)).imag)/t_n)*delta_t)

fourier_call_val = S0*(1/2 + first_integral/np.pi)-np.exp(-r*T)*K*(1/2 + second_integral/np.pi)
print(fourier_call_val)


sigma = .3
gamma = .75
T = [x/12 for x in range(1,13)]
share_prices = {}
mc_share_prices = {}
std_share_prices = {}

for i in range(1,51):
    # Added for monte carlo
    norm_array = norm.rvs(size = 1000*i)

    S_T = S0
    prices = [S_T]
    prices_mc = [S_T]
    std_mc=[0]
    for t in T:
        S_T = S_T*np.exp((risk_free- ((sigma*S_T**(1-gamma))**2/2)*(1/12)) + (sigma*S_T**(1-gamma))*np.sqrt(1/12)*np.random.normal(0,1, i*1000))
        prices.append(S_T)
        #added monte carlo
        S_T_MC = S_T*np.exp((risk_free- ((sigma*S_T**(1-gamma))**2/2)*(1/12)) + (sigma*S_T**(1-gamma))*np.sqrt(1/12)*norm_array)
        prices_mc.append(np.mean(S_T_MC))
        std_mc.append(np.std(S_T_MC))
                            
    share_prices[str(i)] = prices 
    mc_share_prices[str(i)] = prices_mc
    std_share_prices[str(i)]=std_mc
    

for key, value in share_prices.items():
    print(key, np.mean(np.mean(value)))



def path_generator(current_price, risk_free, sigma, gamma, Z, dt):
    """The path_generator function simulates the share price or 
        firm value path
 
    Args: 
        present_price (float/int): initial share price
        riskfree (float/int): risk free rate
        sigma (float/int): share volatility
        Z (float/int): normal random variables
        dt (float/int): term of share price

 
 
    Returns: 
        price (float/int): price of the option
 
    """
    # S_T_MC = S_T*np.exp((risk_free- ((sigma*S_T**(1-gamma))**2/2)*(1/12)) + (sigma*S_T**(1-gamma))*np.sqrt(1/12)*norm_array)
    return current_price * np.exp(np.cumsum(((risk_free - sigma**2/2)*dt + sigma*current_price**(gamma-1)*np.sqrt(dt)*Z), axis=1))


dt = 1/12 # time step
numb = 50 # number of simulations
T = 1
M1 = 1 + T/dt  # the number of time periods (in our case the number of months in a year)

x = np.arange(0, T+dt, dt)

for i in range(1, numb+1):
    Z = norm.rvs(size=1000*i*int(M1))
    Z = Z.reshape(1000*i, int(M1))
    stockvalue = path_generator(S0, risk_free, sigma, gamma, Z, x)


plt.figure(figsize = [11, 6])
plt.plot(x*12, stockvalue[0:20].T)
plt.axhline(S0, c="black")
plt.axhline(S0, c="black")
plt.xlabel('Time (Month)',)
plt.ylabel('Firm value trajectories')
plt.title('Trajectories of stock price in the Merton model')
plt.show()