import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
# plt.style.use('ggplot')
import os
import networkx as nx
import nxviz as nv
import pickle as pkl 

path = 'C:\\Users\\okigboo\\Desktop\\PythonDataScience\\NetworkAnalysis\\timeSeriesGraph'
os.chdir(path)

with open ('Gs.p', 'rb') as f:
    Gs = pkl.load(f)

fig = plt.figure()

# Create a list of the number of edges per month
edge_sizes = [len(g.edges()) for g in Gs]

# Plot edge sizes over time
plt.plot(edge_sizes)
plt.xlabel('Time elapsed from first month (in months).')
plt.ylabel('Number of edges')
plt.show()



def ECDF(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = sorted(data)

    # y-data for the ECDF: y
    y = np.arange(1, n + 1) / n

    return x, y


def ECDF2(data2):
    """Compute ECDF for a one-dimensional array of measurements."""
    data = [float(x) for x in data2]
    # Number of unique points
    x = sorted(set(data))

    # The x of evenly spread values
    xx = np.linspace(x[0], x[-1], len(x))

    # y empty list 
    y = list()

    # iterate over the x evenly spread values
    for i in x:
        ytemp = [s for s in data if s <= i]
        val = len(ytemp) / len(data)
        y.append(val)

    return x, y


# import numpy as np
# from statsmodels.distributions.empirical_distribution import ECDF

# Create a list of degree centrality scores month-by-month
cents = []
for G in Gs:
    cent = nx.degree_centrality(G)
    cents.append(cent)
print(len(cents))
print(sorted(cents[1]))

# Plot ECDFs over time
fig = plt.figure()
for i in range(len(cents)):
    x, y = ECDF2(cents[i].values())
    plt.plot(x, y, label='Month {0}'.format(i+1))
plt.legend()
plt.show()


num_bins = 20
data11 = np.random.randn(10000)
counts, bin_edges = np.histogram(data11, bins=num_bins, normed=True)
cdf = np.cumsum(counts)
plt.plot(bin_edges[1:], cdf/cdf[-1])
plt.show()


x, y = ECDF2(data11)
plt.plot(x, y)
plt.show()




