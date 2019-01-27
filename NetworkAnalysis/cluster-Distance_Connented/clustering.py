import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
# plt.style.use('ggplot')
import os
import networkx as nx
import nxviz as nv
from itertools import combinations


G = nx.karate_club_graph()
nx.info(G)
nx.draw(G, with_labels=True)
plt.show()

""" 
Clustering coefficient: measures the tendency of nodes in a network
to cluster or form 'closed' triangles

Triadic Closure:
The tendency of people who shared common social connection to be
connected. This is measure by looking at open triangles 

this is assessed using clustering

local clustering coefficient of a node

(# of pairs of node's friends who are friends) / (# of all possible pairs of node's friends)


for all possible pairs of node's friends use   eC2 where e is the node's degree
"""




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


# B. Transitivity: This is a measure of number of triangles that are open
# i.e the ratio of number of triangles over the num of open triangles in the network
# use nx.transitivity(Graph)
# tansitivity weighs noed with high degree higher
print(nx.transitivity(G))  # 0.26
        # nx.transitive_closure(G),
        # nx.transitive_reduction(G))