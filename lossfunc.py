import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
plt.style.use('ggplot')

def lossfunc(predProb, observed, clipper=1e-9):
    """The function computes the log loss between the
        predProb and actual values

        : param predProb: the predProb probability that the value is 1
        : param observed: the observed label, either 0 or 1
        : param clipper: the ensure the offset and limited probability close to 1 and 0

        This function provides a steep penalty for prediction that are both wrong
        and confident, i.e when a high probability is given for the incorrect label
    """
    predProb = np.clip(predProb, clipper, 1 - clipper)
    logloss = -1 * np.mean(observed * np.log(predProb) +
                          (1 - observed) * np.log(1 - predProb))
    return logloss

print(lossfunc(predProb=0.9, observed=0))
print(lossfunc(predProb=0.5, observed=1))
print('{:.20f}'.format(lossfunc(predProb=0.85, observed=1)), "=> A")
print('{:.20f}'.format(lossfunc(predProb=0.99, observed=0)), "=> B")
print('{:.20f}'.format(lossfunc(predProb=0.51, observed=0)), "=> C")
