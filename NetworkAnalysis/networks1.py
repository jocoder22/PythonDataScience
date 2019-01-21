from nxviz import ArcPlot
from nxviz import CircosPlot
import pickle
import networkx as nx 
import nxviz as nz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
plt.style.use('ggplot')

filename = 'c:\\Users\\okigboo\\Desktop\\PythonDataScience\\NetworkAnalysis\\ego-twitter.p'

infile = open(filename, 'rb')
myfile = pickle.load(infile)
infile.close()

# G = nx.path_graph(50)
# nx.write_gpickle(G, filename)
G = nx.read_gpickle(filename)

# G = nx.read_gpickle(filename)
print(type(G))


# Create Ghe CircosPlot object: c
c = CircosPlot(G)

# Draw c to the screen
c.draw()
plt.show()


# Create the un-customized ArcPlot object: a
a = ArcPlot(G)

# Draw a to the screen
a.draw()

# Display the plot
plt.show()

# Create the customized ArcPlot object: a2
a2 = ArcPlot(G, node_order='category', node_color='category')
print(type(G.nodes()))
# type(G.edges())
# type(G.node)
# type(G.edge)


