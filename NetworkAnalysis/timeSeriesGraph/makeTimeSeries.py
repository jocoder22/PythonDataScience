from datetime import datetime
import os
import networkx as nx
import nxviz as nv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
# plt.style.use('ggplot')
import pickle as pkl 

path = 'C:\\Users\\okigboo\\Desktop\\PythonDataScience\\NetworkAnalysis\\timeSeriesGraph'
os.chdir(path)

with open ('uci-forum.p', 'rb') as f:
    G = pkl.load(f)

edgelist = list()

for sender, recipient, d in G.edges(data=True):
    edgeinfo = {'sender': sender[1:], 'recipient': recipient[1:]}
    edgeinfo.update(d)
    edgelist.append(edgeinfo)


data = pd.DataFrame(edgelist)
data['year'] = pd.DatetimeIndex(data['date']).year
data['month'] = pd.DatetimeIndex(data['date']).month
data['day'] = pd.DatetimeIndex(data['date']).day
data['hour'] = pd.DatetimeIndex(data['date']).hour
data['minute'] = pd.DatetimeIndex(data['date']).minute
data['second'] = pd.DatetimeIndex(data['date']).second

print(data.head())
print(data.month.unique())
months = range(5, 11)
# Initialize an empty list: Gs
Gs = []
for month in months:
    # Instantiate a new undirected graph: G
    G = nx.Graph()

    # Add in all nodes that have ever shown up to the graph
    G.add_nodes_from(data['sender'])
    G.add_nodes_from(data['recipient'])

    # Filter the DataFrame so that there's only the given month
    df_filtered = data[data['month'] == month]

    # Add edges from filtered DataFrame
    G.add_edges_from(zip(df_filtered['sender'], df_filtered['recipient']))

    # Append G to the list of graphs
    Gs.append(G)

print(len(Gs))

with open ('Gs.p', 'wb') as ff:
    pkl.dump(Gs, ff)


# Instantiate a list of graphs that show edges added: added
added = []
# Instantiate a list of graphs that show edges removed: removed
removed = []
# Here's the fractional change over time
fractional_changes = []
window = 1
i = 0

for i in range(len(Gs) - window):
    g1 = Gs[i]
    g2 = Gs[i + window]

    # Compute graph difference here
    added.append(nx.difference(g2, g1))
    removed.append(nx.difference(g1, g2))

    # Compute change in graph size over time
    fractional_changes.append(
        (len(g2.edges()) - len(g1.edges())) / len(g1.edges()))

# Print the fractional change
print(fractional_changes)


fig = plt.figure()
ax1 = fig.add_subplot(111)

# Plot the number of edges added over time
edges_added = [len(g.edges()) for g in added]
plot1 = ax1.plot(edges_added, label='added', color='orange')

# Plot the number of edges removed over time
edges_removed = [len(g.edges()) for g in removed]
plot2 = ax1.plot(edges_removed, label='removed', color='purple')

# Set yscale to logarithmic scale
ax1.set_yscale('log')
ax1.legend()

# 2nd axes shares x-axis with 1st axes object
ax2 = ax1.twinx()

# Plot the fractional changes over time
plot3 = ax2.plot(fractional_changes, label='fractional change', color='green')

# Here, we create a single legend for both plots
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)
plt.axhline(0, color='green', linestyle='--')
plt.show()
