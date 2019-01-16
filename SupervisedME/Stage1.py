import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
plt.style.use('ggplot')

iris = datasets.load_iris()
print(type(iris))

print(iris.keys())

print(type(iris.data), type(iris.target))

iris.data.shape

iris.target_names

x = iris.data
y = iris.target

df = pd.DataFrame(x, columns=iris.feature_names)

print(df.head())


_ = pd.scatter_matrix(df, c=y, figsize=[8, 8], s=150, marker='.')
plt.pause(3)
plt.clf()

_ = pd.scatter_matrix(df, c=y, figsize=[8, 8], s=150, marker='D')
plt.show()


plt.figure()
sns.countplot(x='education', hue='party', data=df, palette='RdBu')
plt.xticks([0, 1], ['No', 'Yes'])
plt.show()

