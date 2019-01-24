import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
# plt.style.use('ggplot')
import os
import networkx as nx
import nxviz as nv

G = nx.karate_club_graph()

# Robustness: ability to withstand failures or attack
# attacks on the nodes or edges

################ undirected graph
## find the smaller number of node to attack to disconnect the network
print(nx.node_connectivity(G))  # 1 node

#### find the particular node
print(nx.minimum_node_cut(G))  # node 0


# for the edges
print(nx.edge_connectivity(G))

print(nx.minimum_edge_cut(G))

G.remove_node(0)

nx.draw(G, with_labels=True)
plt.show()

nx.draw_spectral(G, with_labels=True)
plt.show()

