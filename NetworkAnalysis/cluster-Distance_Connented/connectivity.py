import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
# plt.style.use('ggplot')
import os
import networkx as nx
import nxviz as nv

G = nx.karate_club_graph()

########################### Undirected graph: 
# Graph is connected is there is a path between pairs of nodes
print(nx.is_connected(G)) # True


#### connected components
print(nx.number_connected_components(G))


####### show the connected components
print(sorted(nx.connected_components(G)))


########### show the component which a node belongs to
print(sorted(nx.node_connected_component(G, 4)))




#########################  Directed Graph: connectivity
# Strongly connected, there is path from a -> b and path form b -> a
# print(nx.is_strongly_connected(G))

# Weakly connected the result if the graph becomes Undirected
print(nx.is_weakly_connected(G))


############ Strongly connected components
print(sorted(nx.strongly_connected_components(G)))


############ Weakly connected components
print(sorted(nx.weakly_connected_components(G)))