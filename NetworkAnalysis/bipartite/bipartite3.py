import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
# plt.style.use('ggplot')
import os
import networkx as nx
import nxviz as nv

url = 'https://assets.datacamp.com/production/course_3556/datasets/american-revolution.csv'

df = pd.read_csv(url, sep=',', skiprows=1,
                 names=["name","StAndrewsLodge","LoyalNine","NorthCaucus",
                        "LongRoomClub","TeaParty","BostonCommittee",
                         "LondonEnemies"])
print(df.head())

node1 = list()
node2 = list()
weight = list()
for i, row in df.iterrows():
        for col in df.columns:
                if row[col] == 1:
                        node1.append(row['name'])
                        node2.append(col)
                        weight.append(1)

df2 = pd.DataFrame({'node1': node1,
                    'node2' : node2,
                    'weight': weight})

print(df2.head())

G = nx.from_pandas_edgelist(df2, 'node1', 'node2', edge_attr=['weight'])

# Assign nodes to 'clubs' or 'people' partitions
for n, d in G.nodes(data=True):
    if '.' in n:
        G.node[n]['bipartite'] = 'people'
    else:
        G.node[n]['bipartite'] = 'clubs'

# Print the edges of the graph
# print(G.edges(data=True))
print(len(G.edges()))
print(len(G.nodes()))
