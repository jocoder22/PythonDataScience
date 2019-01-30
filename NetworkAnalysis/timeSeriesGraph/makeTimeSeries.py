from datetime import datetime
import os
import networkx as nx
import nxviz as nv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
# plt.style.use('ggplot')
import pickle as pkl 

path = 'C:\\Users\\okigboo\\Desktop\\PythonDataScience\\NetworkAnalysis\\timeSeriesGraph'
os.chdir(path)

with open ('uci-forum.p', 'rb') as f:
    G = pkl.load(f)

edgelist = list()

for sender, recipient, d in G.edges(data=True):
    edgeinfo = {'sender': sender[1:], 'recipient': recipient[1:]}
    edgeinfo.update(d)
    edgelist.append(edgeinfo)


data = pd.DataFrame(edgelist)
data['year'] = pd.DatetimeIndex(data['date']).year
data['month'] = pd.DatetimeIndex(data['date']).month
data['day'] = pd.DatetimeIndex(data['date']).day
data['hour'] = pd.DatetimeIndex(data['date']).hour
data['minute'] = pd.DatetimeIndex(data['date']).minute
data['second'] = pd.DatetimeIndex(data['date']).second

print(data.head())
print(data.month.unique())
months = range(5, 11)
# Initialize an empty list: Gs
Gs = []
for month in months:
    # Instantiate a new undirected graph: G
    G = nx.Graph()

    # Add in all nodes that have ever shown up to the graph
    G.add_nodes_from(data['sender'])
    G.add_nodes_from(data['recipient'])

    # Filter the DataFrame so that there's only the given month
    df_filtered = data[data['month'] == month]

    # Add edges from filtered DataFrame
    G.add_edges_from(zip(df_filtered['sender'], df_filtered['recipient']))

    # Append G to the list of graphs
    Gs.append(G)

print(len(Gs))
