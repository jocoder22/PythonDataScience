import os
import numpy as np
import pandas as pd
import networkx as nx
import nxviz as nv
import matplotlib.pyplot as plt
import operator as opp
from itertools import combinations as comb
from math import sqrt
from matplotlib import mlab
from scipy.stats import norm


def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n + 1) / n

    return x, y

def dlist(dd):
    pp = dict()
    for key, value in dd.items():
        if value not in pp:
            pp[value] = key
        elif type(pp[value]) == list:
            pp[value].append(key)
        else:
            pp[value] = [pp[value], key]

    return pp


G = nx.karate_club_graph()

D = dict(G.degree())
DD = list(D.values())
Dv = sorted(set(DD))
Nn = [DD.count(x) for x in Dv]
Nf = [DD.count(x)/float(nx.number_of_nodes(G)) for x in Dv]
n_bins = int(sqrt(len(D)))

Gdata = np.array(DD)
mu, sigma, size = np.mean(Gdata), np.std(Gdata), len(Gdata)
print(mu, sigma, size)


plt.subplot(131)
# Plot histogram of the number of degress
_ = plt.hist(DD, bins=n_bins)

# Label axes
_ = plt.ylabel('count')
_ = plt.xlabel('Node Degree')
plt.title('Simple Histogram')


plt.subplot(132)
# Compute ECDF for Graph degree data: x_vers, y_vers
x_vers, y_vers = ecdf(DD)

# Generate plot
_ = plt.plot(x_vers, y_vers, marker='.', linestyle='-', label='Empirical')


x = np.random.normal(mu, sigma, size=size)

count_, bins  = np.histogram(x, n_bins, density=True)
# Add a line showing the expected distribution.
y = norm.pdf(bins, mu, sigma).cumsum()
y /= y[-1]
plt.plot(bins, y, 'k--', linewidth=1.5, label='Theoretical')


# Label the axes
_ = plt.ylabel('cumulative density')
_ = plt.xlabel('Node Degree')
_ = plt.margins(0.02)
plt.legend(loc='right')
plt.title('Cumulative histograms Normalized')


plt.subplot(133)

values, base = np.histogram(DD, bins=n_bins)
cumulative = np.cumsum(values)

plt.plot(base[:-1], cumulative/cumulative[-1],'k--', c='blue',  label='Empirical\n Non Normalized')
plt.legend(loc='right')

# Label the axes
_ = plt.ylabel('cumulative density')
_ = plt.xlabel('Node Degree')
_ = plt.margins(0.02)
plt.title('Cumulative histograms')

# Show histogram
plt.show()




# Plot Bar chart for relationship between Degree and Degree fraction, and number of nodes
# plt.subplot(121)
Gdat = np.array(Dv)
Dv2 =  Gdat + 0.7
plt.bar(Dv, Nn, width=0.9, label="Nodes in Red")
plt.xlabel('Degree')
plt.ylabel('Number of Nodes')
plt.xticks(Dv)

for key, values in dlist(D).items():
    if type(values) != int:
        word = ""
        for i in values:
            s = str(i)
            word = word+'\n'+s
        plt.text(key, 0.5 , word, color='red', va='bottom', ha='center', fontweight='bold',             label='nodes' )
    else:
        plt.text(key, 0.5, values, color='red', va='bottom', ha='center', fontweight='bold' )

plt.legend(loc='upper right')
plt.show()
# plt.subplot(122)
plt.bar(Dv, Nf)
plt.xlabel('Degree')
plt.ylabel('Fraction of Nodes')
plt.tight_layout()
plt.xticks(Dv)


plt.show()







            
