#!/usr/bin/env python

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
print(Gc.node['f71'])


# Instantiate a new Graph: G
G = nx.Graph()

# Add nodes from each of the partitions
G.add_nodes_from(data['student'], bipartite='student')
G.add_nodes_from(data['forum'], bipartite='forum')

# Add in each edge along with the date the edge was created
for r, d in data.iterrows():
    G.add_edge(d['student'], d['forum'], date=d['date'])
