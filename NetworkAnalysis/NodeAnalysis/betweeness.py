import os
import networkx as nx
import nxviz as nv
import operator

G = nx.karate_club_graph()

# Betweeeness
# Important nodes are once that connect other nodes
bet_c = nx.betweenness_centrality(G, normalized=True, endpoints=False)
print(sorted(((v, k) for k, v in bet_c.items()), reverse=True)[:5])
print(sorted(bet_c.items(), reverse=True, key=operator.itemgetter(1))[:5])
