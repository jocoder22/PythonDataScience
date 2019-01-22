from collections import defaultdict
from itertools import combinations
from nxviz import ArcPlot
from nxviz import CircosPlot
from nxviz.plots import ArcPlot
from nxviz import MatrixPlot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
# plt.style.use('ggplot')
import os
import networkx as nx
import nxviz as nv


""" G = nx.erdos_renyi_graph(n=100, p=0.2, seed=123)
print(len(G.nodes()))
print(len(G.edges()))


# Plot the degree distribution of the collaboration network
plt.hist(list(nx.degree_centrality(G).values()))
plt.show()


plt.hist(list(nx.betweenness_centrality(G).values()))
plt.show()

circ = nv.CircosPlot(G)

circ.draw()
plt.show()


##################### MatrixPlot
G = nx.erdos_renyi_graph(n=15, p=0.6, seed=123)
# Calculate the largest connected component subgraph: largest_ccs
largest_ccs = sorted(nx.connected_component_subgraphs(G),
                     key=lambda x: len(x))[-1]

# Create the customized MatrixPlot object: h
# h = MatrixPlot(graph=largest_ccs, node_grouping='grouping')
h = MatrixPlot(graph=largest_ccs)
# Draw the MatrixPlot to the screen
h.draw()
plt.show()



############################ ArcPlot
# Iterate over all the nodes in G, including the metadata
for n, d in G.nodes(data=True):

    # Calculate the degree of each node: G.node[n]['degree']
    G.node[n]['degree'] = nx.degree(G, n)

# Create the ArcPlot object: a
a = ArcPlot(G, node_order='degree')

# Draw the ArcPlot to the screen
a.draw()
plt.show()



##################### CircosPlot
# Iterate over all the nodes, including the metadata
for n, d in G.nodes(data=True):

    # Calculate the degree of each node: G.node[n]['degree']
    G.node[n]['degree'] = nx.degree(G, n)

# Create the CircosPlot object: c
# c = CircosPlot(G, node_order='degree',
#                node_grouping='grouping', node_color='grouping')
c = CircosPlot(G, node_order='degree')

# Draw the CircosPlot object to the screen
c.draw()
plt.show()
 """


################# Finding Cliques
G = nx.erdos_renyi_graph(n=57, p=0.67, seed=123)

# Calculate the maximal cliques in G: cliques
cliques = nx.find_cliques(G)

# Count and print the number of maximal cliques in G
print(len(list(cliques)))

# Plotting the cliques
# Find the nodes that are part of the largest maximal clique: largest_clique
largest_clique = sorted(nx.find_cliques(G), key=lambda x: len(x))[-1]
print(len(largest_clique))
# Create the subgraph of the largest_clique: G_lc
G_lc = G.subgraph(largest_clique)

# Create the CircosPlot object: c
c = CircosPlot(G_lc)

# Draw the CircosPlot to the screen
c.draw()
plt.show()


################### Recommendation System
# Compute the degree centralities of G: deg_cent
deg_cent = nx.degree_centrality(G)

# Compute the maximum degree centrality: max_dc
max_dc = max(deg_cent.values())

# Find the user(s) that have collaborated the most: prolific_collaborators
prolific_collaborators = [n for n, dc in deg_cent.items() if dc == max_dc]

# Print the most prolific collaborator(s)
print(prolific_collaborators)


# Identify the largest maximal clique: largest_max_clique
largest_max_clique = set(sorted(nx.find_cliques(G), key=lambda x: len(x))[-1])

nodelist = list(largest_max_clique)


# Create a subgraph from the largest_max_clique: G_lmc
G_lmc = G.subgraph(largest_max_clique)
G_lmc = nx.Graph(G_lmc)
# Go out 1 degree of separation
# for node in G_lmc.nodes():
for node in nodelist:
    n_num = len(list(G.neighbors(node)))
    G_lmc.add_nodes_from(G.neighbors(node))
    G_lmc.add_edges_from(zip([node]*n_num, G.neighbors(node)))

# Record each node's degree centrality score
for n in G_lmc.nodes():
    G_lmc.node[n]['degree centrality'] = nx.degree_centrality(G_lmc)[n]

# Create the ArcPlot object: a
a = ArcPlot(G_lmc, node_order='degree centrality')

# Draw the ArcPlot to the screen
a.draw()
plt.show()

 ######################## Recommendation proper
# Initialize the defaultdict: recommended
recommended = defaultdict(int)

# Iterate over all the nodes in G
for n, d in G.nodes(data=True):

    # Iterate over all possible triangle relationship combinations
    for n1, n2 in combinations(G.neighbors(n), 2):

        # Check whether n1 and n2 do not have an edge
        if not G.has_edge(n1, n2):

            # Increment recommended
            recommended[(n1, n2)] += 1

# Identify the top 10 pairs of users
all_counts = sorted(recommended.values())
top10_pairs = [pair for pair, count in recommended.items() if count >
                                                         all_counts[-10]]
print(top10_pairs)



GG = nx.erdos_renyi_graph(n=100, p=0.03, seed=123)
list(nx.connected_component_subgraphs(GG))

for g in list(nx.connected_component_subgraphs(GG)):
    print(len(g.nodes()))
