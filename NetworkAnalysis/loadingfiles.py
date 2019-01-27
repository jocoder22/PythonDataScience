import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
# plt.style.use('ggplot')
import os
import networkx as nx
import nxviz as nv

path = 'C:\\Users\\okigboo\\Desktop\\PythonDataScience\\NetworkAnalysis'
os.chdir(path)


###### loading from Adjacency list
G = nx.read_adjlist('adjacencylist.txt', nodetype=int)
print(G.edges())

###### loading from Edge list
GG = nx.read_edgelist('edgelist.txt', data=[('Weight', int)])
print('Printing GG...')
print(GG.edges(data=True))


############## load from pandas DataFrame
df = pd.DataFrame([(0, 1,  4, 'friends'),
                   (0, 2,  3, 'coworker'),
                   (0, 3,  2, 'siblings'),
                   (0, 5,  6, 'siblings'),
                   (1, 3,  2, 'friends'),
                   (1, 6,  5, 'friends'),
                   (3, 4,  3, 'coworker'),
                   (5, 4,  1, 'coworker'),
                   (5, 8,  6, 'coworker'),
                   (4, 7,  2, 'coworker'),
                   (8, 9,  1, 'friends')],
                   columns=['source', 'target', 'Weight', 'relation'])

print(df.head())
Gf = nx.from_pandas_edgelist(df, edge_attr=True)
print(Gf.edges(data=True))
print(type(df))


df2 = df.copy()
df2.rename({"source": "a", "target": "b"}, axis='columns', inplace=True)
print(df2.head())
Gf2 = nx.from_pandas_edgelist(df2, 'a', 'b', edge_attr=['Weight', 'relation'])
print(Gf2.edges(data=True))


G = nx.MultiGraph()
G.add_node('A', role='manager')
G.add_edge('A', 'B', relation='friend')
G.add_edge('A', 'C', relation='business partner')
G.add_edge('A', 'B', relation='classmate')
G.node['A']['role'] = 'team member'
G.node['B']['role'] = 'engineer'


Gk = nx.karate_club_graph()
nx.draw(Gk, with_labels=True)
plt.show()




nodelist = pd.read_csv('https://gist.githubusercontent.com/brooksandrew/f989e10af17fb4c85b11409fea47895b/raw/a3a8da0fa5b094f1ca9d82e1642b384889ae16e8/nodelist_sleeping_giant.csv')

edgelist = pd.read_csv('https://gist.githubusercontent.com/brooksandrew/e570c38bcc72a8d102422f2af836513b/raw/89c76b2563dbc0e88384719a35cba0dfc04cd522/edgelist_sleeping_giant.csv')

T =  nx.from_pandas_edgelist(edgelist, 'node1', 'node2', edge_attr=True)

for n in nodelist['id']:
    for col in nodelist.columns[1:]:
        pp = int(nodelist[col][nodelist['id'] == n].values)
        T.node[n][col] = pp

T.nodes['y_rt']

nx.draw(T, with_labels=True)
plt.show()


