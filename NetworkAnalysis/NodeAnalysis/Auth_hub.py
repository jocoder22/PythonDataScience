import os
import numpy as np
import pandas as pd
import networkx as nx
import nxviz as nv
from nxviz import CircosPlot
import matplotlib.pyplot as plt
import operator as opp
from itertools import combinations as comb

# np.random.seed(8943)
np.random.seed(83)
# np.random.seed(183)
# np.random.seed(3948)

G = nx.karate_club_graph()

# Assessing Authority score and Hub scores
# Authority scores in based on number of incoming links = 1
# Hub scores is based on number of outgoing links = 0
# ahscores = nx.hits(G)
# print(len(ahscores))

# print(sorted(ahscores[0].items(), reverse=True, key=opp.itemgetter(1))[:5])
# print(sorted(ahscores[1].items(), reverse=True, key=opp.itemgetter(1))[:5])



############################# Directed Graph
Gd = nx.DiGraph()
Gd.add_edges_from([('A', 'B'), ('C', 'A'), ('A', 'E'), ('G', 'A'), ('A', 'N'), ('B', 'C'), ('D', 'B'), ('B', 'E'), ('C', 'D'), ('E', 'C'), ('D', 'E'), ('E', 'D'), ('F', 'G'), ('I', 'F'), ('J', 'F'),
                  ('H', 'G'), ('I', 'G'), ('G', 'J'), ('I', 'H'), ('H', 'I'), ('I', 'J'), ('J', 'O'), ('O', 'J'), ('K', 'M'), ('K', 'L'), ('O', 'K'), ('O', 'L'), ('N', 'L'), ('L', 'M'), ('N', 'O')])

ahscoresd = nx.hits(Gd, max_iter=1900)
print(len(ahscoresd))
print(len(Gd.nodes()))

""" 
print(sorted(ahscoresd[0].items(), reverse=True, key=opp.itemgetter(1))[:5])
print(sorted(ahscoresd[1].items(), reverse=True, key=opp.itemgetter(1))[:5])

"""
for k,v in ahscoresd[0].items():
    Gd.nodes[k]['Hub'] = v 

for k,v in ahscoresd[1].items():
    Gd.nodes[k]['Auth'] = v 

nodesizeHub = np.array(list(ahscoresd[0].values())) * 12000
np.clip(nodesizeHub, 250.0, np.max(nodesizeHub), out=nodesizeHub)
nodesizeAuth = np.array(list(ahscoresd[1].values())) * 18000
np.clip(nodesizeAuth, 250.0, np.max(nodesizeAuth), out=nodesizeAuth)

nodecolorHub = np.array(list(ahscoresd[0].values()))  * 1259
# nodecolorHub = np.linspace(1.0, 255.0, len(Gd.nodes()))
np.clip(nodecolorHub, 20.0, np.max(nodecolorHub), out=nodecolorHub)
nodecolorAuth = np.array(list(ahscoresd[1].values())) * 1000
np.clip(nodecolorAuth, 20.0, np.max(nodecolorAuth), out=nodecolorAuth)


for k,vv in ahscoresd[1].items():
    v = vv * 155
    vn = vv * 7800
    Gd.nodes[k]['nodecolorAuth'] = v
    Gd.nodes[k]['nodesizeAuth'] = vn  

# print(nodecolorHub)
# print(nodesizeHub)

plt.subplot(221)
nx.draw(Gd, with_labels=True, node_size=nodesizeHub,
        node_color=nodecolorHub, cmap="coolwarm")

plt.title('Showing the HITS Hub scores')


plt.subplot(222)
nx.draw(Gd, with_labels=True, node_size=nodesizeAuth,
        node_color=nodecolorAuth, cmap="coolwarm")

plt.title('Showing the HITS Authority scores')

# plt.show()




plt.subplot(223)
nx.draw_circular(Gd, with_labels=True, node_size=nodesizeHub,
        node_color=nodecolorHub, cmap="coolwarm")

plt.title('Showing the HITS Hub scores')


# plt.show()
plt.subplot(224)
nx.draw_spectral(Gd, with_labels=True, node_size=nodesizeHub,
        node_color=nodecolorHub, cmap="coolwarm")

plt.title('Showing the HITS Hub scores')



circ = nv.CircosPlot(Gd, node_labels=True, node_order="Auth", 
                node_color="Auth", node_size='nodesizeAuth')
circ.draw()
plt.title('Showing the HITS Authority scores')
plt.show()

print(Gd.nodes['A'])
print(nodesizeAuth)
