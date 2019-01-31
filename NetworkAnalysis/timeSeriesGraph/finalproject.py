#!/usr/bin/env python

from nxviz import CircosPlot
from datetime import datetime, timedelta
import os
import networkx as nx
import nxviz as nv
import pickle as pkl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
# plt.style.use('ggplot')
import pickle as pkl


path = 'C:\\Users\\okigboo\\Desktop\\PythonDataScience\\NetworkAnalysis\\timeSeriesGraph'
os.chdir(path)

with open('uci-forum.p', 'rb') as f:
    Gc = pkl.load(f)

edgelist = list()

for sender, recipient, d in Gc.edges(data=True):
    edgeinfo = {'student': sender, 'forum': recipient}
    edgeinfo.update(d)
    edgelist.append(edgeinfo)

data = pd.DataFrame(edgelist)

print(data.head())

print(Gc.edges(['f71' , 's630'], data=True))
print(Gc.edges(data=True)[0:5])
print(Gc.node['f71'])


# Instantiate a new Graph: G
G = nx.Graph()

# Add nodes from each of the partitions
G.add_nodes_from(data['student'], bipartite='student')
G.add_nodes_from(data['forum'], bipartite='forum')

# Add in each edge along with the date the edge was created
for r, d in data.iterrows():
    G.add_edge(d['student'], d['forum'], date=d['date'])


# Get the student partition's nodes: student_nodes
student_nodes = [n for n, d in G.nodes(
    data=True) if d['bipartite'] == 'student']

# Create the students nodes projection as a graph: G_students
G_students = nx.bipartite.projected_graph(G, nodes=student_nodes)

# Calculate the degree centrality using nx.degree_centrality: dcs
dcs = nx.degree_centrality(G_students)

# Plot the histogram of degree centrality values
plt.hist(list(dcs.values()))
plt.yscale('log')
plt.show()


# Get the forums partition's nodes: forum_nodes
forum_nodes = [n for n, d in G.nodes(data=True) if d['bipartite'] == 'forum']

# Create the forum nodes projection as a graph: G_forum
G_forum = nx.bipartite.projected_graph(G, nodes=forum_nodes)

# Calculate the degree centrality using nx.degree_centrality: dcs
dcg = nx.degree_centrality(G_forum)

# Plot the histogram of degree centrality values
plt.hist(list(dcg.values()))
plt.yscale('log')
plt.show()


year = 2011
month = 11
day1 = 10
day2 = 6
date1 = datetime(year, month, day1)
date2 = datetime(year, month, day2)
print(date1 > date2)

date3 = datetime(2011, 11, 10, 0, 0)
print(date3)
days = 4
td = timedelta(days)
print(dt)

# Instantiate a new graph: G_sub
G_sub = nx.Graph()

# Add nodes from the original graph
G_sub.add_nodes_from(G.nodes(data=True))

# Add edges using a list comprehension with one conditional on the edge dates, that the date of the edge is earlier than 2004-05-16.
G_sub.add_edges_from([(u, v, d) for u, v, d in G.edges(
    data=True) if d['date'] < datetime(2004, 5, 16)])


# Compute degree centrality scores of each node
dcs = nx.bipartite.degree_centrality(G, nodes=forum_nodes)

for n, d in G_sub.nodes(data=True):
    G_sub.node[n]['dc'] = dcs[n]

# Create the CircosPlot object: c
c = CircosPlot(G_sub, node_color='bipartite',
               node_grouping='bipartite', node_order='dc')

# Draw c to screen
c.draw()

# Display the plot
plt.show()
