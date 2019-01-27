import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import nxviz as nv


G = nx.karate_club_graph()
""" 
1. most degree :  in_degree_centrality
2. average proximity to other nodes:  closeness_centrality
3. tend to connect to other nodes : shortest path: betweeness_centrality
4. Page rank: ranks pages based on number of links to the node
5. Hits authority score: important nodes are referrenced most by other nodes

Network Centrality
 -- influential nodes
 -- Information spread
 -- hubs in transport network 
 -- important pages on the web
 -- Nodes that prevent network from breaking


Common Centrality measures
    Degree Centrality
    Closeness Centrality
    Between Centrality
    Load Centrality
    Katz Centrality
    Percolation Centrality

Undirected networks   -    Degree
Directed networks     -    in-degree or out-degree
 """


def degCent(Graph, directed=False, i=False):
    try:
        if directed:
            if i:
                didegree = nx.in_degree_centrality(Graph)
                print(sorted(((v, k) for k, v in didegree.items()), reverse=True)[:5])
            else:
                didegreeout = nx.out_degree_centrality(Graph)
                print(sorted(((v, k) for k, v in didegreeout.items()), reverse=True)[:5])

            
        else:
            undegree = nx.degree_centrality(Graph)
            print(sorted(((v, k) for k, v in undegree.items()), reverse=True)[:5])


    except:
        print('make sure you are using the right graph and function')


################################# Degree Centrality
# important nodes has more nodes connected to them
############################### Undirected networks 
degree_c = nx.degree_centrality(G)

d_view = [(v, k) for k, v in degree_c.items()]
print(sorted(d_view, reverse=True)[:5])

print(sorted(((v, k) for k, v in degree_c.items()), reverse=True))



def sterm(position):
    return position[1]

d_view2 = [(k, v) for k, v in degree_c.items()]
print(sorted(d_view2, reverse=True, key=sterm))
print(sorted(d_view2, reverse=True, key=lambda x: x[1]))


Gd = nx.DiGraph()
Gd.add_edges_from([('A', 'B'), ('C', 'A'), ('A', 'E'), ('G', 'A'), ('A', 'N'), ('B', 'C'), ('D', 'B'), ('B', 'E'), ('C', 'D'), ('E', 'C'), ('D', 'E'), ('E', 'D'), ('F', 'G'), ('I', 'F'), ('J', 'F'),
                  ('H', 'G'), ('I', 'G'), ('G', 'J'), ('I', 'H'), ('H', 'I'), ('I', 'J'), ('J', 'O'), ('O', 'J'), ('K', 'M'), ('K', 'L'), ('O', 'K'), ('O', 'L'), ('N', 'L'), ('L', 'M'), ('N', 'O')])

###################### Directed network 
indegree_c = nx.in_degree_centrality(Gd)
print(indegree_c)
print(sorted(((v, k) for k, v in indegree_c.items()), reverse=True)[:5])

outdegree_c = nx.out_degree_centrality(Gd)
degCent(Gd, directed=True)



############################## close Centrality
# important nodes are close to other nodes
############################# undirected network

close_c = nx.closeness_centrality(G)


############ Directed network 
close_cd = nx.closeness_centrality(Gd, wf_improved=False)
print(close_cd['M'])

close_cd2 = nx.closeness_centrality(Gd, wf_improved=True)
print(close_cd2['M'])


print(sorted(((v, k) for k, v in close_cd2.items()), reverse=True)[:5])
nx.draw(Gd, with_labels=True)
plt.show()

nx.draw_kamada_kawai(Gd, with_labels=True)
plt.show()

