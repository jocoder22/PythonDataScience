import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

url = 'https://datadryad.org/resource/doi:10.5061/dryad.g6g3h'
# Create bee swarm plot
_ = sns.swarmplot('year', 'beak_depth', data=df)

# Label the axes
_ = plt.xlabel('year')
_ = plt.ylabel('beak depth (mm)')

# Show the plot
plt.show()
