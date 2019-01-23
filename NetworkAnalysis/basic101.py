from nxviz.plots import CircosPlot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
# plt.style.use('ggplot')
import os
import networkx as nx
import nxviz as nv
""" """ 
G = nx.MultiDiGraph()

G.add_edge('John', 'Ana', weight=3,
           relation='siblings', key='one', color='red')
G.add_edge('David', 'Ana',  weight=4,
           relation='cousins', key='two', color='green')
G.add_edge('Bob', 'Ana', weight=6, relation='friends', key='one', color='red')
G.add_edge('Ana', 'Bob', weight=3, relation=[
           'neighbors', 'cowokers'], key='one', color='black')
G.add_edge('Ana', 'Joe', weight=5, relation='friends', key='one', color='red')
G.add_edge('Joe', 'David', weight=2,
           relation='friends', key='two', color='orange')
G.add_edge('Bob', 'Mary', weight=5,
           relation='coworkers', key='two', color='yellow')
G.add_edge('Bob', 'Mary', weight=4,
           relation='siblings', key='one', color='blue')
G.add_edge('John', 'Bob', weight=6,
           relation='cousins', key='two', color='pink')
G.add_edge('Joe', 'Mary', weight=4,
           relation='siblings', key='one', color='purple')

 
G.add_nodes_from(['Ana', 'Mary'], gender='Female')
G.add_nodes_from(['Joe', 'David', 'Bob', 'John'], gender='Male')
G.add_nodes_from(['Joe', 'David', 'Mary'], grouping='Catholics')
G.add_nodes_from(['Ana', 'Bob', 'John'], grouping='Budists')
print(G.nodes(data=True))

print(G.node['Ana']['gender'])

G.size()
G.degree()
G.order()
G.edges()
print(G['Ana']['Bob']['one']['relation'])
print(G.edges(data=True))
print(G.edges(data='relation'))
 

nx.draw(G, with_labels=True, node_color=[G.degree(v) for v in G])
plt.pause(2)
plt.clf()

nx.draw_random(G, with_labels=True)
plt.pause(2)
plt.clf()

for n, d in G.nodes(data=True):

    # Calculate the degree of each node: G.node[n]['degree']
    G.node[n]['degree'] = nx.degree(G, n)
    

nx.draw_circular(G, with_labels=True, node_color=[G.degree(v) for v in G], edge_color = 'grey')
edge_labels = dict([((u, v,), d['degree'])
                    for u, v, d in G.edges(data=True)])
nx.draw_networkx_edge_labels(G, edge_labels=edge_labels)
plt.pause(5)
plt.clf()

nx.draw_spectral(G, with_labels=True)
plt.pause(2)
plt.close()


for n, d in G.nodes(data=True):

    # Calculate the degree of each node: G.node[n]['degree']
    G.node[n]['degree'] = nx.degree(G, n)

circ = nv.CircosPlot(G, node_order='degree',
                     node_grouping='grouping',
                     node_color='gender',
                        node_labels=True,
                     # ["beginning", "middle", "end"]
                    #  group_label_position="beginning",
                        group_label_color=True
                         )

circ.draw()
plt.show() 

