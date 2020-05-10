#!/usr/bin/env python
# We will be pricing a vanilla European call option on a single stock under the following conditions:
# o Continuously-compounded interest rate, 𝑟, of 6%
# o Initial stock price, 𝑆0, of $100
# o Stock volatility, 𝜎, 30%
# o Strike price, 𝐾, of $110
# o Maturity time, 𝑇, of one year
# As per usual, we make all the assumptions of the Black-Scholes model.




# import necessary modules
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt



def anal_option_prices(current_price, risk_free, sigma, term, strike_price, current_time=0, type="call"):
    """The Ana_option_prices function calculate both analytical price for either call or put option
 
    Args: 
        present_price (float/int): initial share price
        riskfree (float/int): risk free rate
        sigma (float/int): share volatility
        Z (float/int): normal random variables
        T (float/int): term of share price
        type (str): type of option, "call" or "put"
        strike_price (float/int) : strike pricke
 
    Returns: 
        price (float/int): price of the option
 
    """   
    
    # calculate d1 and d2
    d1_numerator = np.log(current_price/strike_price) 
    d11_numerator = (risk_free + sigma**2/2) * (T - current_time)
    d1_denominator = sigma * np.sqrt(T - current_time)

    d1 = (d1_numerator + d11_numerator) / d1_denominator
    d2 =  d1 - d1_denominator


    if type == "call":
        analytic_price = current_price*norm.cdf(d1) - (norm.cdf(d2)*strike_price*np.exp(-risk_free * (T - current_time)))

    else:
        analytic_price = -current_price*norm.cdf(-d1) + (norm.cdf(-d2)*strike_price*np.exp(-risk_free * (T - current_time)))

        
    return analytic_price



# define characteristic functions
def c_M1_t(s0, r, sigma, T, t):
  """
  
  
  """
  
  s_ij = 1j*t*(np.log(s0) + (r - sigma**2/2)*T)
  
  sigma_t = (sigma**2)* T * (t**2/2)
  
  return np.exp(s_ij - sigma_t)


def c_M2_t(s0, r, sigma, T, t):
  """
  
  
  """
  
  sigma_ij = np.exp(1j*t*sigma**2*T)
  
  M1_t = c_M1_t(s0, r, sigma, T, t)
  
  return sigma_ij*M1_t



def fourier_option_prices(s0, r, sigma, T, K):
  """
  
  
  
  """
  # technique for approximating integral value (using areas of rectangles)
  t_max = 20
  N = 100
  k_log = np.log(K)

  # calculating delta 
  delta_t = t_max/N
  t_range = np.linspace(1,N, N)
  t_n = (t_range - 1/2)* delta_t

  cm1t = c_M2_t(s0, r, sigma, T, t_n)
  cm2t = c_M1_t(s0, r, sigma, T, t_n)

  s0_integral = sum((((np.exp(-1j*t_n*k_log)*cm1t).imag)/t_n)*delta_t)
  k_integral = sum((((np.exp(-1j*t_n*k_log)*cm2t).imag)/t_n)*delta_t)  
  
  s0_part = s0*(1/2 + s0_integral/np.pi)
  k_part = np.exp(-r*T)*K*(1/2 + k_integral/np.pi)
  
  return s0_part - k_part


def get_npi(b2, b1, d, c, n):
  """
  
  
  """

  npi_d = np.pi*n*(d-b1)/(b2-b1)
  npi_c = np.pi*n*(c-b1)/(b2-b1)
  npi_2 = np.pi*n/(b2-b1)
  
  return npi_d, npi_2, npi_c

  
def upsilon_n(b2, b1, d, c, n):
  """
  
  
  """
  
  a, b, cc = get_npi(b2, b1, d, c, n)
  
  val_one = (np.cos(a)*np.exp(d) - np.cos(cc)*np.exp(c))
  val_two = (b*(np.sin(a)*np.exp(d) - np.sin(cc)*np.exp(c)))
  
  return (val_one + val_two) / (1 + (b**2))
  
  
def psi_n(b2, b1, d, c, n):

  """
  
  
  """
  a, b, cc = get_npi(b2, b1, d, c, n) 
  
  if n == 0:
    return d - c
  
  else:
    return 1/b * (np.sin(a) - np.sin(cc))
  
  

def v_n(K, b2, b1, n):
  """
  
  
  """
  
  
  up_silon = upsilon_n(b2, b1, b2, 0, n)
  
  psi_ = psi_n(b2, b1, b2,  0, n)
  
  return 2*K*(up_silon - psi_)/(b2 - b1)


def logchar_func(u, s0, r, sigma, K, T):
  """
  
  
  """
  
  s_ij1 = 1j*u*(np.log(s0/K) + (r - sigma**2/2)*T)
              
  sigma_t1 = (sigma**2) * T * (u**2)/2
  
  return np.exp(s_ij1 - sigma_t1)




def call_price(NN, s0, sigma, r, K, T, b2, b1):
  """
  
  
  """
  
  vn = v_n(K, b2, b1, 0)
  log_char = logchar_func(0, s0, r, sigma, K, T)
  
  price = vn * log_char / 2
  
  for n in range(1,NN):
    b = np.pi*n/(b2-b1)
    vnn = v_n(K, b2, b1, n)
    log_charn = logchar_func(b, s0, r, sigma, K, T)
    exp_n = np.exp(-1j*b*b1)
    
    price = price +  log_charn * exp_n * vnn
  
  
  return price.real*np.exp(-r * T)


# share specific information
s0 = 100
r = 0.06
sigma = 0.3


# option specific information
K = 110
T = 1
k_log = np.log(K)


analytic_callprice = anal_option_prices(s0, r, sigma, T, K, type="call")
fourier_call = fourier_option_prices(s0, r, sigma, T, K)

print(analytic_callprice, fourier_call)


# b1, b2 for call
c1 = r
c2 = T*sigma**2
c4 = 0
L = 10

b1 = c1 - L * np.sqrt(c2 - np.sqrt(c4))
b2 = c1 + L * np.sqrt(c2 - np.sqrt(c4))


# calculate COS for various N
COS_callprice = np.zeros(50)


for i in range(1,51):
  COS_callprice[i-1] = call_price(i, s0, sigma, r, K, T, b2, b1)
  
# plotting the results
plt.plot(COS_callprice)
plt.plot([analytic_callprice]*50)
plt.xlabel("N")
plt.ylabel("Call price")
plt.show()


# plot the log absolute error
plt.plot(np.log(np.absolute(COS_callprice - analytic_callprice)))
plt.xlabel("N")
plt.ylabel("Log absolute error")
plt.show()


#Share info
S0 = 100
sigma = 0.3
T = 1
r=0.06

#Algorithm info
N = 2**10
delta = 0.25
alpha = 1.5

def log_char(u):
    return np.exp(1j*u*(np.log(S0)+(r-sigma**2/2)*T)-sigma**2*T*u**2/2)

def c_func(v):
    val1 = np.exp(-r*T)*log_char(v-(alpha+1)*1j)
    val2 = alpha**2+alpha-v**2+1j*(2*alpha+1)*v
    return val1/val2

n = np.array(range(N))
delta_k = 2*np.pi/(N*delta)
b = delta_k*(N-1)/2

log_strike = np.linspace(-b,b,N)

x = np.exp(1j*b*n*delta)*c_func(n*delta)*(delta)
x[0] = x[0]*0.5
x[-1] = x[-1]*0.5

#I used the in-built fft function to minimise typing here and
#    respond to the question quickly, otherwise, of course,
#    the full example in the notes first defines the complete function then uses it.
xhat = np.fft.fft(x).real 

fft_call = np.exp(-alpha*log_strike)*xhat/np.pi

#call price
d_1 = (np.log(S0/np.exp(log_strike))+(r+sigma**2/2)*T)/(sigma*np.sqrt(T))
d_2 = d_1 - sigma*np.sqrt(T)
analytic_callprice = S0*norm.cdf(d_1)-np.exp(log_strike)*np.exp(-r*(T))*norm.cdf(d_2)

#Multiple things to be noted here about the plotting below. I have labelled the
#    lines as LINE 1 - 6 for easy reference in this explanation.

#NUMBER 1. The form we have been used to of the plot function plots a data set of our own
#    on the y-axis against an automatic number line on the x-axis, as used in the two cases 
#    on page 46 of the notes, for example, in the from plt.plot(y). However, to plot two
#    data sets of our own, with one on the x-axis and one on the y-axis, we use the form:
#    plt.plot(x,y). The set to go on the x-axis goes first, then the set to go on the
#    y-axis goes second.


#NUMBER 2. The variable log_strike holds the values of ln(strike).
#    Therefore, to plot against the strike itself, we have to do the inverse of
#    ln(strike), which is e^strike. This is written here in the code as np.exp(log_strike).


#LINE 1 uses the concepts of NUMBER 1 and NUMBER 2 above, to plot e^strike against
#    the analytical call prices. LINE 2 does the same thing, plotting e^strike against
#    the FFT call price estimates.


#NUMBER 3. Temporarily disable LINE 3 below, by putting # in front of it, then run this code.
#    If you followed NUMBER 1 and NUMBER 2 above, you would get the correct graph.
#    But as you can see, it would not look at all like the graph in the notes. You would notice
#    on the x-axis that the values go beyond 250000. Referring back to page 46 of the M4 notes,
#    the automatic number line on the x-axis takes values up to 50 because there are 50 values
#    in the array COS_callprice. Here in this FFT code, there are 1024 values in log_strike, which
#    means there are also 1024 values of e^strike, the largest being about 282000.
#    So the automatic x-axis goes up to that value. To limit the x-axis to 100, as has been
#    done for the FFT example in the notes, we use LINE 3. You can enable it again now by
#    removing the # and running the code again. The axis function
#    takes, in order, (x-axisMinimum, x-axisMaximum, y-axisMinimum, y-axisMaximum).
#    So LINE 3 has a minimum of 0 and maximum of 100 on the x-axis, so only part of the
#    original graph is shown, and it is the part we need. Now it looks more like the 
#    example in the notes! 
    
plt.plot(np.exp(log_strike), analytic_callprice) #LINE 1
plt.plot(np.exp(log_strike), fft_call) #LINE 2
plt.axis([0,100,0,100]) #LINE 3
plt.xlabel("Strike")
plt.ylabel("Call Price")
plt.show() #LINE 4

#The other plot in the notes is more straightforward. The absolute error
#    is found by subtracting the analytical cal prices from the fft estimates of the prices,
#    then plotting the absolute vlues of the differences, against the logarithms of the strikes.
#    Remember, it matters which one you put first in the plot function.The first one goes on the
#    x-axis, and the second will go on the y-axis. LINE 5 plots the graph, similar to the notes.
#    Hope this helps!

plt.plot(log_strike, np.absolute(fft_call-analytic_callprice)) #LINE 5
plt.xlabel("Log-Strike")
plt.ylabel("Absolute Error")
plt.show()




 




