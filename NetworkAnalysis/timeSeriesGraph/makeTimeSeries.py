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

with open ('uci-forum.p', 'rb') as f:
    G = pkl.load(f)

edgelist = list()

for sender, recipient, dt in G.edges(data=True):
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
    
