import os

# Import plotting modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

path = 'C:\\Users\\Jose\\Desktop\\PythonDataScience\\datavisualization\\seaborn'

os.chdir(path)

# irisurl = "http://aima.cs.berkeley.edu/data/iris.csv"
# iris_data = pd.read_csv(irisurl, sep=',', decimal='.', header=None,
#                         names=['sepal_length', 'sepal_width', 'petal_length',
#                                'petal_width', 'target'])


iris_data = pd.read_csv('iris.csv')

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
plt.pause(2)
plt.clf()




def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n + 1) / n

    return x, y


# Compute ECDF for versicolor data: x_vers, y_vers
x_vers, y_vers = ecdf(versicolor_petal_length)

# Generate plot
_ = plt.plot(x_vers, y_vers, marker='.',linestyle='none')

# Label the axes
_ = plt.ylabel('cumulative density')
_ = plt.xlabel('petal length ( cm )')
_ = plt.margins(0.02)


# Display the plot
plt.pause(2)
plt.close()



nheads = 0
n = 100000
for _ in range(n):
    heads = np.random.random(size=4) < 0.5
    countHead = np.sum(heads)
    if countHead == 4:
        nheads += 1

print(nheads/n)



# Compute mean and standard deviation: mu, sigma
mu = np.mean(belmont_no_outliers)
sigma = np.std(belmont_no_outliers)

# Sample out of a normal distribution with this mu and sigma: samples
samples = np.random.normal(mu, sigma, size=10000)

# Get the CDF of the samples and of the data
x_theor, y_theor = ecdf(samples)
x, y = ecdf(belmont_no_outliers)

# Plot the CDFs and show the plot
_ = plt.plot(x_theor, y_theor)
_ = plt.plot(x, y, marker='.', linestyle='none')
_ = plt.xlabel('Belmont winning time (sec.)')
_ = plt.ylabel('CDF')
plt.show()
