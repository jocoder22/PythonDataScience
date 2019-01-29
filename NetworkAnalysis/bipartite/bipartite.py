cdimport os
import networkx as nx
import nxviz as nv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
# plt.style.use('ggplot')

from networkx.algorithms import bipartite

N = nx.Graph()
N.add_nodes_from(['A', 'B', 'C', 'D', 'E', 'F', 'G'], bipartite=0)
N.add_nodes_from(['ManU', 'Chelsea', 'Westham'], bipartite=1)

N.add_edges_from([('A', 'ManU'), ('C', 'ManU'),
                  ('B', 'ManU'), ('E', 'ManU'), ('D', 'ManU'),
                  ('B', 'Chelsea'), ('E', 'Chelsea'), ('F', 'ManU'),
                  ('G', 'Westham'), ('C', 'Chelsea'), ('A', 'Chelsea'),
                  ('D', 'Chelsea'), ('A', 'Westham'),
                  ('F', 'Westham'), ('C', 'Westham'), ('E', 'Westham')])

nx.draw(N)
plt.show()
print(bipartite.is_bipartite(N))
print(bipartite.sets(N))


p0 = set(bipartite.sets(N)[0])
proj_0 = bipartite.projected_graph(N, p0)
nx.draw(proj_0, with_labels=True)
plt.title('Fans Projection')
plt.show()


p1 = set(bipartite.sets(N)[1])
proj_1 = bipartite.projected_graph(N, p1)
nx.draw(proj_1, with_labels=True)
plt.title('Football Team Projection')
plt.show()





p11 = set(bipartite.sets(N)[1])
proj_11 = bipartite.weighted_projected_graph(N, p11)
# nx.draw(proj_11, with_labels=True)



# chose the layout
pos = nx.random_layout(proj_11)
nx.draw_networkx(proj_11, pos, with_labels=True, node_size=800, node_color='pink',
                 alpha=0.7, edge_color='grey', width=1)
plt.axis('off')
edge_labels = dict([((u, v,), d['weight'])
                    for u, v, d in proj_11.edges(data=True)])
nx.draw_networkx_edge_labels(proj_11, pos, edge_labels=edge_labels)
plt.title('Football Team Projection with Edge weights')
print(proj_11.edges(data=True))

plt.show()


gg = nx.Graph()
gg.add_edges_from([('C', 'F'), ('C', 'E'), ('B', 'F'), ('B', 'E'),
                    ('A', 'E'), ('A', 'G')])

nx.draw(gg)
plt.show()


print(bipartite.is_bipartite(gg))
print(bipartite.sets(gg))

FanBase = {}
for u, v, d in N.edges(data=True):
    # Calculate the degree of each node: G.node[n]['degree']
    d['degree'] = nx.degree(N, v)
    fans = nx.degree(N, u)
    N.add_edge(u, v , Fans=fans)
    if v not in FanBase:
        FanBase[v] = d['degree']




print(N.edges(data=True))
print(FanBase)
