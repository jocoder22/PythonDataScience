import os
import numpy as np
import pandas as pd
import networkx as nx
import nxviz as nv
import matplotlib.pyplot as plt
import operator as opp
from itertools import combinations as comb
from math import sqrt


def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n + 1) / n

    return x, y


G = nx.karate_club_graph()

D = dict(G.degree())
DD = list(D.values())
Dv = sorted(set(DD))
Nn = [DD.count(x) for x in Dv]
Nf = [DD.count(x)/float(nx.number_of_nodes(G)) for x in Dv]
n_bins = int(sqrt(len(D)))


plt.subplot(121)
# Plot histogram of the number of degress
_ = plt.hist(DD, bins=n_bins)

# Label axes
_ = plt.ylabel('count')
_ = plt.xlabel('Node Degree')


plt.subplot(122)
# Compute ECDF for Graph degree data: x_vers, y_vers
x_vers, y_vers = ecdf(DD)

# Generate plot
_ = plt.plot(x_vers, y_vers, marker='.', linestyle='-')

# Label the axes
_ = plt.ylabel('cumulative density')
_ = plt.xlabel('Node Degree')
_ = plt.margins(0.02)

# Show histogram
plt.show()


# Plot Bar chart for relationship between Degree and Degree fraction, and number of nodes
plt.subplot(121)
plt.bar(Dv, Nn)
plt.xlabel('Degree')
plt.ylabel('Number of Nodes')

plt.subplot(122)
plt.bar(Dv, Nf)
plt.xlabel('Degree')
plt.ylabel('Fraction of Nodes')

plt.show()



