import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
# plt.style.use('ggplot')
import os
import networkx as nx
import nxviz as nv


G = nx.karate_club_graph()
nx.info(G)
nx.draw(G, with_labels=True)
plt.show()

""" Triadic Closure:
The tendency of people who shared common social connection to be
connected. This is measure by looking at open triangles 

this is assessed using clustering

local clustering coefficient of a node

(# of pairs of node's friends who are friends) / (# of all possible pairs of node's friends)


for all possible pairs of node's friends use   eC2 where e is the node's degree
"""

from itertools import combinations


def p_pairs(n):
    total = len(list(combinations(list(range(n)), 2)))
    return total

print(p_pairs(G.degree(15)))

############### Local clustering --- using nx.clustering(Graph, node)
print(nx.clustering(G, 23))  # 0.4
print(nx.clustering(G, 11))  # 0.0



############### Global clustering, measures the average number of open triangles
# A. Average of all local clustering -- using nx.average_clustering(Graph)
print(nx.average_clustering(G)) # 0.57
