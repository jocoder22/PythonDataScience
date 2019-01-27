import os
import networkx as nx
import nxviz as nv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator

G = nx.DiGraph()
G.add_edges_from([('A', 'B'), ('C', 'A'), ('A', 'E'), ('G', 'A'), ('A', 'N'), ('B', 'C'), ('D', 'B'), ('B', 'E'), ('C', 'D'), ('E', 'C'), ('D', 'E'), ('E', 'D'), ('F', 'G'), ('I', 'F'), ('J', 'F'),
                  ('H', 'G'), ('I', 'G'), ('G', 'J'), ('I', 'H'), ('H', 'I'), ('I', 'J'), ('J', 'O'), ('O', 'J'), ('K', 'M'), ('K', 'L'), ('O', 'K'), ('O', 'L'), ('N', 'L'), ('L', 'M'), ('N', 'O')])


prank = nx.pagerank(G, alpha=0.9)
# print(prank)
# print(sorted(prank.items(), reverse=True, key=operator.itemgetter(1))[:5])

for k,v in prank.items():
    G.nodes[k]['weight'] = v 


# nodes = [n for n, d in T.nodes(data=True) if d['occupation'] == 'celebrity']
print(G)
# nodesize = [d.values() for n, d in G.nodes(data=True)]
nodesize = np.array(list(prank.values())) * 40000
nodecolor = np.array(list(prank.values())) * 100
print(nodesize)

nx.draw(G, with_labels=True, node_size=nodesize,
        node_color=nodecolor, cmap="rainbow")
plt.show()


nx.draw_random(G, with_labels=True, node_size=nodesize,
            node_color=nodecolor,  cmap='Paired')
plt.show()


nx.draw_circular(G, with_labels=True, node_size=nodesize,
                 node_color=nodecolor,  cmap='plasma')
plt.show()

""" nx.draw_shell(G, with_labels=True, node_size=nodesize,
              node_color=nodecolor,  cmap='Accent') """
""" plt.show()

nx.draw_spring(G, with_labels=True, node_size=nodesize,
               node_color=nodecolor,  cmap='Paired')
plt.show() """

nx.draw_kamada_kawai(G, with_labels=True, node_size=nodesize,
                     node_color=nodecolor,  cmap='coolwarm')
plt.show()


print(sorted(prank.items(), reverse=True, key=operator.itemgetter(1))[:5])