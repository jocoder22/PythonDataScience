import os
import numpy as np
import pandas as pd
import random
from itertools import product

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from printdescribe import print2, changepath

# path = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
path2 = r"D:\PythonDataScience\Hyperparameter_Tuning"

# data = pd.read_excel(path,header=1, index_col=0)
# data = data.rename(columns={'default payment next month':"default"})

"""

with changepath(path2):
    # data.to_csv("credit_default.csv",  index=False,  compression='gzip')
    data = pd.read_csv("credit_default.csv",  compression='gzip')

print2(data.head())

X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-1], data.iloc[:,-1], stratify=data.iloc[:,-1], test_size=0.3)
print2(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Create lists for max_depth, learn_rate and min_samples_leaf
max_depth_list = list(range(1,66))
min_samples_leaf_list = list(range(20, 70))
learning_list = np.linspace(0.01,2,100)



# Combination list
combinations_list = [list(x) for x in product(max_depth_list, learning_list, min_samples_leaf_list)]
print2(len(combinations_list))

model_rfc = GradientBoostingClassifier(n_estimators=100)

# Create the parameter grid
param_grid = {'max_depth': max_depth_list,
             'min_samples_leaf': min_samples_leaf_list,
             'learning_rate':learning_list} 

# Create a RandomizedSearchCV object
random_rfclass = RandomizedSearchCV(
    estimator = model_rfc,
    param_distributions = param_grid, 
    n_iter = 100, verbose=1,
    scoring='accuracy', n_jobs=4, cv = 2, refit=True, return_train_score = True)


random_rfclass.fit(X_train, y_train)
results = pd.DataFrame(random_rfclass.cv_results_)

"""

with changepath(path2):
    # results.to_csv('results3.csv',  index=False,  compression='gzip')
    results = pd.read_csv('results.csv',  compression='gzip')

print2(results.head(), results.columns)

# newdata = results.loc[:,['param_min_samples_leaf', 'param_max_depth', 'param_learning_rate', 'mean_test_score', 'mean_train_score']]

# print2(newdata.head())


def visualize_first():
  for name in results_df.columns[0:2]:
    plt.clf()
    plt.scatter(results_df[name],results_df['accuracy'], c=['blue']*500)
    plt.gca().set(xlabel='{}'.format(name), ylabel='accuracy', title='Accuracy for different {}s'.format(name))
    plt.gca().set_ylim([0,100])
    x_line = 20
    if name == "learn_rate":
      	x_line = 1
    plt.axvline(x=x_line, color="red", linewidth=4)
    plt.show()
    
    
