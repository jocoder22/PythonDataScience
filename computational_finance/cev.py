#Importing libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ncx2


#Variable declaration
S0 = 100
sigma = 0.3
gamma = 0.75
r = 0.1
T = 3
#Strikes to test volatility
test_strikes = np.linspace(80,120,41)


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

analytic_callprice = S0*norm.cdf(d_1_stock)-K*np.exp(-r*(T))*norm.cdf(d_2_stock)


#Call price under CEV
z = 2 + 1/(1-gamma)

def C(t,K):
    kappa = 2*r/(sigma**2*(1-gamma)*(np.exp(2*r*(1-gamma)*t)-1))
    x = kappa*S0**(2*(1-gamma))*np.exp(2*r*(1-gamma)*t)
    y = kappa*K**(2*(1-gamma))
    return S0*(1-ncx2.cdf(y,z,x))-K*np.exp(-r*t)*ncx2.cdf(x,z-2,y)

#Share specific information
S0 = 100
v0 = 0.06
kappa = 9
theta = 0.06
r = 0.1
sigma = 0.3
rho = -0.4

#Variable declaration
S0 = 100
sigma = 0.3
gamma = 0.75
r = 0.1
T = 3



#Call Option specific information
K = 105
T = 3
k_log = np.log(K)

#Approximation information
t_max = 30
N = 100


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


first_integral = sum((((np.exp(-1j*t_n*k_log)*adj_char(t_n)).imag)/t_n)*delta_t)
second_integral = sum((((np.exp(-1j*t_n*k_log)*log_char(t_n)).imag)/t_n)*delta_t)

fourier_call_val = S0*(1/2 + first_integral/np.pi)-np.exp(-r*T)*K*(1/2 + second_integral/np.pi)
print(fourier_call_val, analytic_callprice, **sp)

