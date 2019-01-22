import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
# plt.style.use('ggplot')
import os
import networkx as nx
import nxviz as nv

G = nx.MultiDiGraph()

G.add_edge('John', 'Ana', weight=3,
           relation='siblings', key='one', color='red')
G.add_edge('Ana', 'David', weight=4,
           relation='cousins', key='one', color='green')
G.add_edge('Bob', 'Ana', weight=6, relation='friends', key='one', color='red')
G.add_edge('Ana', 'Bob', weight=3, relation=[
           'neighbors', 'cowokers'], key='one', color='black')
G.add_edge('Ana', 'Joe', weight=5, relation='friends', key='one', color='red')
G.add_edge('Joe', 'David', weight=2,
           relation='friends', key='two', color='blue')


G.size()
G.degree()
G.order()
G.edges()
print(G['Ana']['Bob']['one']['relation'])
print(G.edges(data=True))

nx.draw(G, with_labels=True)
plt.show()
