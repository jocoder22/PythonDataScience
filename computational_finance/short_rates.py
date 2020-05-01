import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import random

# set market parameters
r0 =  0.05
alpha = 0.2
b = 0.08
sigma = 0.025


#  define useful functions
def vasicek_mean(r, t1, t2):
  """The vasicek_mean function calculate the mean of of short rate at t2 given the
      short rate at t1 < t2 using the vasicek method
      
      Input:
        r(float): interest rate
        t1(int/float) : time at t1
        t2(int/float) : time at t2
        
       Output:
          result(float) : the vasicek mean
  
  """
  r0_discounted = r * np.exp(-alpha*(t2-t1))
  b_discounted = b * (1 - np.exp(-alpha*(t2-t1)))
 
  result = r0_discounted + b_discounted

  return result


def vasicek_var(t1,t2):
  """The vasicek_var function calculate the variance of short rate at t2 given the
      short rate at t1 < t2 using the vasicek method
      
      Inputs:
        t1(int/float) : time at t1
        t2(int/float) : time at t2
        
      Output:
        result(float) : vasicek variance
  
  """
 
  term1 = sigma**2/(2 * alpha)
  term2 = 1 - np.exp(-2*alpha * (t2-t1))
  
  result =  term1* term2

  return result


# simulate interest rate paths
np.random.seed(0)

nyears = 10
simulations = 10

t = np.array(range(0,nyears+1))

z = norm.rvs(size = [simulations, nyears])
r_sim = np.zeros([simulations, nyears])
r_sim[:,0] = r0
vasicek_mean_vector = np.zeros(nyears+1)


for i in range(nyears):
  r_sim[:,i+1] = vasicek_mean(r_sim[:,i],t[i], t[i+1]) + np.sqrt(vasicek_var(t[i], t[i+t])) * z[:,i]
  
  
s_mean = r0 * np.exp(-alpha*t) + b*(1-np.exp(-alpha*t))


# plot results
_graph = np.ones(r_sim.shape)*t
plt.plot(np.transpose(_graph), np.transpose(r_sim*100), 'r')
plt.plot(t, s_mean*100)
plt.xlabel("Year")
plt.ylabel("Short Rate")
plt.show()
