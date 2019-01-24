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


# Robust networks have large minimum_edge_cut and mininum_node_cut numbers


G = nx.DiGraph()
G.add_edges_from([('A', 'B'), ('C', 'A'), ('A', 'E'), ('G', 'A'), ('A', 'N'), ('B', 'C'), ('D', 'B'), ('B', 'E'), ('C', 'D'), ('E', 'C'), ('D', 'E'), ('E', 'D'), ('F', 'G'), ('I', 'F'), ('J', 'F'),
                  ('H', 'G'), ('I', 'G'), ('G', 'J'), ('I', 'H'), ('H', 'I'), ('I', 'J'), ('J', 'O'), ('O', 'J'), ('K', 'M'), ('K', 'L'), ('O', 'K'), ('O', 'L'), ('N', 'L'), ('L', 'M'), ('N', 'O')])

######## Directed Graphs
# To print all the path between two nodes
print(sorted(nx.all_simple_paths(G, 'A', 'L')))

# How many nodes to remove
print(nx.node_connectivity(G, 'A', 'L'))


# which nodes to remove
print(nx.minimum_node_cut(G, 'A', 'L'))


# How edges to remove
print(nx.edge_connectivity(G, 'A', 'L'))


# which edges to remove
print(nx.minimum_edge_cut(G, 'A', 'L'))


# Blocking from H to O
# How many nodes to remove
print('Nodes to remove from H to 0: ', nx.node_connectivity(G, 'H', 'O'))

# which nodes to remove
print('Which Nodes to remove from H to 0: ', nx.minimum_node_cut(G, 'H', 'O'))

# How edges to remove
print('Edges to remove from H to 0: ', nx.edge_connectivity(G, 'H', 'O'))

# which edges to remove
print('Which Edges to remove from H to 0: ', nx.minimum_edge_cut(G, 'H', 'O'))
