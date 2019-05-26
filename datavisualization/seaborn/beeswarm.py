import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

path = 'C:\\Users\\Jose\\Desktop\\PythonDataScience\\datavisualization\\seaborn'

os.chdir(path)

# Set default Seaborn style
sns.set()


# irisurl = "http://aima.cs.berkeley.edu/data/iris.csv"
# df = pd.read_csv(irisurl, sep=',', decimal='.', header=None,
#                         names=['sepal_length', 'sepal_width', 'petal_length',
#                                'petal_width', 'target'])

# df.to_csv('iris.csv')
df = pd.read_csv('iris.csv')



# url = 'https://assets.datacamp.com/production/course_1549/datasets/2008_all_states.csv'

# election = pd.read_csv(url, sep=',')

# election.to_csv('election.csv')
election = pd.read_csv('election.csv')

print(election.head())
print(election.columns) 
#  Index(['state', 'county', 'total_votes', 'dem_votes', 'rep_votes',
#        'other_votes', 'dem_share', 'east_west'],
#       dtype='object') 

print(election.info())


opfl = election.query("state in ['FL', 'PA', 'OH']") # returns complete data

# Create bee swarm plot with Seaborn's default settings
_ = sns.swarmplot(x=opfl.state, y=opfl.dem_share, data=opfl)

# Label the axes
_ = plt.xlabel('states')
_ = plt.ylabel('Democratic votes (percentage of total votes)')

# Show the plot
plt.pause(4)
plt.clf()




# Create bee swarm plot with Seaborn's default settings
_ = sns.swarmplot(x=df['target'], y=df['petal_length'], data=df)

# Label the axes
_ = plt.xlabel('species')
_ = plt.ylabel('petal length (cm)')

# Show the plot
plt.pause(4)
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

# selected = election['state', ]
oh = election.dem_share[election.state == 'OH']
pa = election.dem_share[election.state == 'PA']
fl = election.dem_share[election.state == 'FL']

xoh, yoh = ecdf(oh)
xpa, ypa = ecdf(pa)
xfl, yfl = ecdf(fl)

_ = plt.plot(xoh, yoh , marker='.', linestyle='none')
_ = plt.plot(xpa, ypa , marker='.', linestyle='none')
_ = plt.plot(xfl, yfl , marker='.', linestyle='none')

_ = plt.ylabel('ECDF')
_ = plt.xlabel('Percentag of voters')

_ = plt.legend(labels=('Ohio', 'Pennsylvania','Florida'), loc='lower right')

plt.pause(6)
plt.close()



# Make a scatter plot
_ = plt.plot(versicolor_petal_length, versicolor_petal_width, marker='.', linestyle='none')


# Label the axes

_ = plt.ylabel('versicolor petal width')
_ = plt.xlabel('versicolor petal length')

# Show the result
plt.show()
