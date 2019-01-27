import os
import numpy as np
import pandas as pd
import networkx as nx
import nxviz as nv
import matplotlib.pyplot as plt
import operator as opp
from itertools import combinations as comb

G = nx.karate_club_graph()

# Assessing Authority score and Hub scores
# Authority scores in based on number of incoming links
# Hub scores is based on number of outgoing links
print(nx.hits(G))