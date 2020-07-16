# import required modules
import numpy as np
import pandas as pd
from printdescribe import print2

# initialize prices at s0 and s1
s0 = np.array([1,100,200])
s1 = np.array([[1,110,200],
              [1,100,220],
              [1,90,180]])

# compute inverse of s1
si = np.linalg.inv(s1)
sm = np.matrix(s1)
s2 = sm.I

# compute state prices
tau = s0 @ si
taua_transpose = s0.dot(si)

pf = pd.DataFrame(s1)
print2(s0,s1,si,s2, tau,taua_transpose, np.transpose(tau), pf)

# initialize the probabilities
prob = np.array([0.25, 0.5, 0.25])

# compute Xstar, the unique solution
Xstar = 100*prob / tau
X_star = Xstar.reshape(-1,1)


# compute strategy with optimal utility
x_star = si @ X_star
x_star2 = si @ Xstar
print2(x_star, x_star2, Xstar)
