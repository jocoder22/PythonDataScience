#!/usr/bin/env python
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from zipfile import ZipFile
from io import BytesIO
import re
import networkx as nx
from nxviz import ArcPlot
from nxviz.plots import CircosPlot
from random import choice
from functools import reduce


# plt.style.use('ggplot')

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sp = '\n\n'

url2 = 'https://assets.datacamp.com/production/repositories/2162/datasets/94f2356652dc9ea8f0654b5e9c29645115b6e77f/chapter_4.zip'


try:
    response = requests.get(url2, timeout=3)
    response.raise_for_status()
except requests.exceptions.HTTPError as errh:
    print("Http Error:", errh)
except requests.exceptions.ConnectionError as errc:
    print("Error Connecting:", errc)
except requests.exceptions.Timeout as errt:
    print("Timeout Error:", errt)
except requests.exceptions.RequestException as err:
    print("OOps: Something Else ...", err)


# unzip the content
zipp = ZipFile(BytesIO(response.content))

# Dsiplay files names in the zip file
mylist = [filename for filename in zipp.namelist()]


# load the file into pandas dataframe
data = pd.read_csv(zipp.open(mylist[5]))


# Compile regex
p = re.compile(r'[\w.]+')

data['source'] = data.From.apply(lambda x: p.search(x).group())
data['target'] = data.To.apply(lambda x: p.search(x).group())


data['target']  = data.target.shift(-2)
data.dropna(inplace=True)


# graph = nx.DiGraph()

# graph.add_edges_from(data)

graph = nx.from_pandas_edgelist(data, 'source', 'target',
                                create_using=nx.DiGraph())

sizes = [val[1]*210 for  val in graph.degree()]


# Calculate the degree of each node: G.node[n]['degree']
for n, d in graph.nodes(data=True):
    graph.node[n]['degree'] = nx.degree(graph, n)
    graph.node[n]["sizes"] = choice([3, 8, 5, 13, 20, 15])

print(graph.number_of_nodes(), graph.number_of_edges(), sep=sp)

# # Create Ghe CircosPlot object: c
# c = CircosPlot(graph)

# # Draw c to the screen
# c.draw()
# plt.show()


# nx.draw(graph, node_size=sizes,
#         node_color=sizes, cmap="rainbow")
# plt.show()

# # Create random layout positions
# pos = nx.random_layout(graph)
# pos2 = nx.circular_layout(graph)
# # Draw the network
# nx.draw_networkx(graph, pos,
#                  with_labels=False,
#                  node_size=sizes,
#                  node_color=sizes, alpha=0.7,
#                  arrowsize=2, linewidths=0,
#                  cmap="plasma")
# plt.axis('off')
# plt.show()


# nx.draw_networkx(graph, pos2,
#                  with_labels=False,
#                  node_size=sizes,
#                  node_color=sizes, alpha=0.7,
#                  arrowsize=2, linewidths=0,
#                  cmap="plasma")
# plt.axis('off')
# plt.show()

# # Create the un-customized ArcPlot object: a
# a = ArcPlot(graph)

# # Draw a to the screen
# a.draw()

# # Display the plot
# plt.show()


# circ = CircosPlot(graph, node_order='degree',
#                   node_color='degree',
#                   node_size='degree')

# circ.draw()
# plt.show()


# for n, d in G.nodes(data=True):
#     G.node[n]["sizes"] = choice([3,8,5,13,20,15])

# d = CircosPlot(
#     G,
#     node_grouping="class",
#     group_order="default",
#     node_color="class",
#     node_order="class",
#     node_size='sizes',
#     node_labels=False,
#     group_label_position="middle",
#     group_label_color=True,
# )
# d.draw()
# plt.show()

column_names = ['Staff_Name', 'degree']
betweeness = pd.DataFrame(list(nx.betweenness_centrality(graph).items()),
                    columns=column_names)
centrality = pd.DataFrame(
    list(nx.in_degree_centrality(graph).items()), columns=column_names)
degree_in = pd.DataFrame(list(graph.in_degree()), columns=column_names)

# # Merge the two DataFrames on screen name
# ratio = centrality.merge(degree_in, betweeness, on='Staff_Name',
#                         suffixes=('_cent', '_deg', '_bet'))

dfs = [betweeness, centrality, degree_in]
suf = ('_bet', '_cent')
ratio = reduce(lambda left, right: pd.merge(left, right, on='Staff_Name'), dfs)

# ratio = pd.concat([betweeness, centrality, degree_in], axis=1)

print(ratio.head(), ratio.columns, sep=sp)
# Calculate the ratio
ratio['ratio'] = ratio['degree_x'] / ratio['degree_y']

# Exclude any staff with less than 5 emails
ratio = ratio[ratio['degree'] >= 5]

# Print out first five with highest ratio
print(ratio.sort_values('ratio', ascending=False).head(), sep=sp*2)
print(ratio.sort_values('degree', ascending=False).head(), sep=sp*2)
print(ratio.sort_values('degree_x', ascending=False).head())
