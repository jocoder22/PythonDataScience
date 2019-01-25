from random import choice
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
# plt.style.use('ggplot')
import os
import networkx as nx
import nxviz as nv
from nxviz.plots import ArcPlot


# Path: sequence of nodes connnect by edges
G = nx.karate_club_graph()
"""
nx.draw_random(G, with_labels=True)
# plt.pause(2)
# plt.clf()
plt.show()


nx.draw_spectral(G, with_labels=True)
# plt.pause(2)
# plt.close()
plt.show()

nx.draw_circular(G, with_labels=True)
plt.show()

# Path length: number of step or edges from one node to another
# Distance is the shortest path between one node to another
print(nx.shortest_path(G, 1, 14))  # list nodes on the path [1, 2, 32, 14]
print(nx.shortest_path_length(G, 1, 14))  # 3


# finding the distance between a node and the rest in the network
# using BreathFirst search
T = nx.bfs_tree(G, 13)
nx.draw_circular(T, with_labels=True)
plt.show()


nx.draw_spectral(T, with_labels=True)
plt.show()

print(T.edges())
print(nx.shortest_path_length(G, 13)) # dict of nodes and their distance from node 13
 """

##### avearge distance of node pairs in the network
print(nx.average_shortest_path_length(G))  # 2.41


# max possible distance between 2 nodes called Diameter
print(nx.diameter(G))


######### Eccentricity measures the largest distance between a node and other nodes
# returns a dict with nodes and their diameter
print(nx.eccentricity(G, 4))


############## Radius of a graph is the minimuim eccentricity of the graph
############## Diameter is the maximium eccentricity of the graph
print(nx.radius(G))


###### Periphery of a graph is(are) set(s) of node whose eccentricity == diamenter
print(nx.periphery(G))

# Iterate over all the nodes in G, including the metadata
for n, d in G.nodes(data=True):

    # Calculate the degree of each node: G.node[n]['degree']
    G.node[n]['degree'] = nx.degree(G, n)
    G.node[n]["class"] = choice(["one", "two", "three"])
    G.node[n]['distance'] = nx.eccentricity(G, n)
    # G.node[n]['center'] = nx.radius(G)


a = ArcPlot(G, node_color="degree", node_order="distance",
                node_labels=True)
# Draw the ArcPlot to the screen
# print(G.node[3])
a.draw()
plt.axis('off')
plt.show()


###### Cemter nodes has eccentricity == radius
print(nx.center(G))  # sensitive to outlier node


# pos = nx.get_node_attributes(G, 'class')
pos = nx.kamada_kawai_layout(G)
nx.draw_networkx(G, pos)
plt.axis('off')
plt.show()
