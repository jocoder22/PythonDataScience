# Import plotting modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

irisurl = "http://aima.cs.berkeley.edu/data/iris.csv"
iris_data = pd.read_csv(irisurl, sep=',', decimal='.', header=None,
                        names=['sepal_length', 'sepal_width', 'petal_length',
                               'petal_width', 'target'])

vcl = iris_data.target == 'versicolor'
versicolor_petal_length = iris_data.petal_length[vcl]

# Set default Seaborn style
sns.set()

n_data = len(versicolor_petal_length)

# Number of bins is the square root of number of data points: n_bins
n_bins = np.sqrt(n_data)

# Convert number of bins to integer: n_bins
n_bins = int(n_bins)

# Plot histogram of versicolor petal lengths
_ = plt.hist(versicolor_petal_length, bins=n_bins)

# Label axes
_ = plt.ylabel('count')
_ = plt.xlabel('petal length (cm)')

# Show histogram
plt.show()

