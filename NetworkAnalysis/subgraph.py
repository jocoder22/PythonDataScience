#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
import networkx as nx
import nxviz as nz
from nxviz import ArcPlot
from nxviz import CircosPlot
# plt.style.use('ggplot')
from itertools import combinations

T = nx.erdos_renyi_graph(n=20, p=0.2, seed=10)

nx.draw(T, with_labels=True)
plt.show()
nodes = list(T.neighbors(10))

nodes.append(10)
G_10 = T.subgraph(nodes)

nx.draw(G_10, with_labels=True)
plt.show()