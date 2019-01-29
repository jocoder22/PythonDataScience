import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
# plt.style.use('ggplot')
import os
import networkx as nx
import nxviz as nv
from bipartite2 import get_nodes_from_partition
# from bfunctions import get_nodes_from_partition

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

n1 = 0
n2 = 0
# Assign nodes to 'clubs' or 'people' partitions
for n, d in G.nodes(data=True):
    if '.' in n:
        G.node[n]['bipartite'] = 'people'
        n1 += 1
    else:
        G.node[n]['bipartite'] = 'clubs'
        n2 += 1

# Print the edges of the graph
# print(G.edges(data=True))
# print(G.nodes())

print(len(G.edges()))
print(len(G.nodes()))
print(n1, n2)

# Prepare the nodelists needed for computing projections: people, clubs
people = [n for n in G.nodes() if G.node[n]['bipartite'] == 'people']
clubs = [n for n, d in G.nodes(data=True) if d['bipartite'] == 'clubs']

# Compute the people and clubs projections: peopleG, clubsG
peopleG = nx.bipartite.projected_graph(G, people)
clubsG = nx.bipartite.projected_graph(G, clubs)


# Plot the degree centrality distribution of both node partitions from the original graph
plt.figure()
original_dc = nx.bipartite.degree_centrality(G, people)
plt.hist(list(original_dc.values()), alpha=0.5)
plt.yscale('log')
plt.title('Bipartite degree centrality')
plt.show()


# Plot the degree centrality distribution of the peopleG graph
plt.figure()
people_dc = nx.degree_centrality(peopleG)
plt.hist(list(people_dc.values()))
plt.yscale('log')
plt.title('Degree centrality of people partition')
plt.show()

# Plot the degree centrality distribution of the clubsG graph
plt.figure()
clubs_dc = nx.degree_centrality(clubsG)
plt.hist(list(clubs_dc.values()))
plt.yscale('log')
plt.title('Degree centrality of clubs partition')
plt.show()


# Get the list of people and list of clubs from the graph: people_nodes, clubs_nodes
people_nodes = get_nodes_from_partition(G, 'people')
clubs_nodes = get_nodes_from_partition(G, 'clubs')

# Compute the biadjacency matrix: bi_matrix
bi_matrix = nx.bipartite.biadjacency_matrix(
    G, row_order=people_nodes, column_order=clubs_nodes)

# Compute the user-user projection: user_matrix
user_matrix = bi_matrix @ bi_matrix.T

print(user_matrix)
