from collections import defaultdict
import os
import networkx as nx
import nxviz as nv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
# plt.style.use('ggplot')


def get_nodes_from_partition(Graph, partition):
    # Initialize an empty list for nodes to be returned
    nodes = []
    # Iterate over each node in the graph G
    for n, d in Graph.nodes(data=True):
        # Check that the node belongs to the particular partition
        if Graph.node[n]['bipartite'] == partition:
            # If so, append it to the list of nodes
            nodes.append(n)
    return nodes


# # Print the number of nodes in the 'projects' partition
# print(len(get_nodes_from_partition(G, 'projects')))

# # Print the number of nodes in the 'users' partition
# print(len(get_nodes_from_partition(G, 'users')))


# # Get the 'users' nodes: user_nodes
# user_nodes = get_nodes_from_partition(G, 'users')

# # Compute the degree centralities: dcs
# dcs = nx.degree_centrality(G)

# # Get the degree centralities for user_nodes: user_dcs
# user_dcs = [dcs[n] for n in user_nodes]

# # Plot the degree distribution of users_dcs
# plt.yscale('log')
# plt.hist(user_dcs, bins=20)
# plt.show()


# project_nodes = get_nodes_from_partition(G, 'projects')

# # Compute the degree centralities: dcs
# dcs = nx.degree_centrality(G)

# # Get the degree centralities for project_nodes: project_dcs
# project_dcs = [dcs[n] for n in project_nodes]

# # Plot the degree distribution of project_dcs
# plt.yscale('log')
# plt.hist(project_dcs, bins=20)
# plt.show()


def shared_partition_nodes(G, node1, node2):
    # Check that the nodes belong to the same partition
    assert G.node[node1]['bipartite'] == G.node[node2]['bipartite']

    # Get neighbors of node 1: nbrs1
    nbrs1 = G.neighbors(node1)
    # Get neighbors of node 2: nbrs2
    nbrs2 = G.neighbors(node2)

    # Compute the overlap using set intersections
    overlap = set(nbrs1).intersection(nbrs2)
    return overlap


# # Print the number of shared repositories between users 'u7909' and 'u2148'
# print(len(shared_partition_nodes(G, 'u7909', 'u2148')))


def user_similarity(G, user1, user2, proj_nodes):
    # Check that the nodes belong to the 'users' partition
    assert G.node[user1]['bipartite'] == 'users'
    assert G.node[user2]['bipartite'] == 'users'

    # Get the set of nodes shared between the two users
    shared_nodes = shared_partition_nodes(G, user1, user2)

    # Return the fraction of nodes in the projects partition
    return len(shared_nodes) / len(proj_nodes)


# # Compute the similarity score between users 'u4560' and 'u1880'
# project_nodes = get_nodes_from_partition(G, 'projects')
# similarity_score = user_similarity(G, 'u4560', 'u1880', project_nodes)

# print(similarity_score)


def most_similar_users(G, user, user_nodes, proj_nodes):
    # Data checks
    assert G.node[user]['bipartite'] == 'users'

    # Get other nodes from user partition
    user_nodes = set(user_nodes)
    user_nodes.remove(user)

    # Create the dictionary: similarities
    similarities = defaultdict(list)
    for n in user_nodes:
        similarity = user_similarity(G, user, n, proj_nodes)
        similarities[similarity].append(n)

    # Compute maximum similarity score: max_similarity
    max_similarity = max(similarities.keys())

    # Return list of users that share maximal similarity
    return similarities[max_similarity]


# user_nodes = get_nodes_from_partition(G, 'users')
# project_nodes = get_nodes_from_partition(G, 'projects')

# print(most_similar_users(G, 'u4560', user_nodes, project_nodes))


def recommend_repositories(G, from_user, to_user):
    # Get the set of repositories that from_user has contributed to
    from_repos = set(G.neighbors(from_user))
    # Get the set of repositories that to_user has contributed to
    to_repos = set(G.neighbors(to_user))

    # Identify repositories that the from_user is connected to that the to_user is not connected to
    return from_repos.difference(to_repos)


# # Print the repositories to be recommended
# print(recommend_repositories(G, 'u7909', 'u2148'))
