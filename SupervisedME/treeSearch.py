from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import randint
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
plt.style.use('ggplot')

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data'

columnsName = ['party', 'handicapped-infants', 'water-project-cost-sharing',
               'adoption-of-the-budget-resolution', 'physician-fee-freeze',
               'el-salvador-aid', 'religious-groups-in-schools', 'satellite',
               'aid-to-nicaraguan-contras', 'missile', 'immigration',
               'synfuels-corporation-cutback', 'education', 'superfund-right-to-sue',
               'crime', 'duty-free-exports', 'exportafrica']


df = pd.read_csv(url, names=columnsName, sep=',')

mapper = {'n': 0, 'y': 1, 'democrat': 0, 'republican': 1}

# df[df == '?'] = np.nan
for colnames in df.columns:
    df[colnames] = df[colnames].str.replace('?', 'n')
    df[colnames] = df[colnames].map(mapper)

y = df['party'].values
X = df.drop('party', axis=1).values

# Setup the parameters and distributions to sample from: param_dist
param_dist = {"max_depth": [3, None],
              "max_features": randint(1, 9),
              "min_samples_leaf": randint(1, 9),
              "criterion": ["gini", "entropy"]}

# Instantiate a Decision Tree classifier: tree
tree = DecisionTreeClassifier()

# Instantiate the RandomizedSearchCV object: tree_cv
tree_cv = RandomizedSearchCV(tree, param_dist, cv=5)

# Fit it to the data
tree_cv.fit(X, y)

# Print the tuned parameters and score
print("Tuned Decision Tree Parameters: {}".format(tree_cv.best_params_))
print("Best score is {}".format(tree_cv.best_score_))
