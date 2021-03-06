#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Import different modules for using with the notebook
from IPython.display import display
from IPython.display import Image
from IPython.display import HTML

from printdescribe import print2

# download the data
iris = datasets.load_iris()

data = iris.data
labels = iris.target
labelnames = iris.target_names


RANDOMSTATE = 42
FIGSIZE = (12, 8)

# create train/test split using 25% test size
X_train, X_test, y_train, y_test = train_test_split(data, labels,
                                                    test_size=0.30,
                                                    random_state=RANDOMSTATE)

# Fit LDA
lda = LDA(n_components=2)
lda.fit(X_train,  y_train)

# Plot the percent explained
plt.plot(range(0, 2), lda.explained_variance_ratio_)
plt.ylabel('Explained Variance')
plt.xlabel('Principal Components')
plt.title('Explained Variance Ratio')
plt.show()

# Bar chart percent explained
pe_var = np.round(lda.explained_variance_ratio_*100, 1)
ldalabels = ["LDA"+str(i) for i in range(1,len(pe_var)+1)]
plt.bar(ldalabels,pe_var)
plt.ylabel("Percentage of Explained Variance")
plt.xlabel("Pricipal Component")
plt.title("Scree Plot")
plt.show()

# vectors = lda.components_.round(3)
# print2(vectors)

# # show percentage made up of each variable
# for i in range(1,len(pe_var)+1):
#     print(f'PC {i} effects = {str(dict(zip(labelnames[:], vectors[i-1])))}')
    
# Perform LDA
lda_result = lda.transform(X_test)

# set the markers and colors
markers = ["*","o", "+"]
colors = ["r", "y", "k"]
labels = y_test

# plot the real data
for i, label in enumerate(np.unique(labels)):
    plt.scatter(lda_result[:,0][labels==label], lda_result[:,1][labels==label], 
                c=colors[i], marker=markers[i], lw=0.5, edgecolors='k', label=labelnames[i])
plt.title("The projection onto 2 LCA components")
plt.legend(loc='best', shadow=False, scatterpoints=3)
plt.show();
