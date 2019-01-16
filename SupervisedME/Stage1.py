import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

plt.style.use('ggplot')

# iris = datasets.load_iris()
# print(type(iris))

# print(iris.keys())

# print(type(iris.data), type(iris.target))

# iris.data.shape

# iris.target_names

# x = iris.data
# y = iris.target

# df = pd.DataFrame(x, columns=iris.feature_names)

# print(df.head())


# _ = pd.plotting.scatter_matrix(df, c=y, figsize=[8, 8], s=150, marker='.')
# plt.pause(1)
# plt.clf()

# _ = pd.plotting.scatter_matrix(df, c=y, figsize=[8, 8], s=150, marker='D')
# plt.pause(1)
# plt.clf()


url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/voting-records/house-votes-84.data'

columnsName = ['party', 'handicapped-infants', 'water-project-cost-sharing',
               'adoption-of-the-budget-resolution', 'physician-fee-freeze',
               'el-salvador-aid', 'religious-groups-in-schools', 'satellite',
               'aid-to-nicaraguan-contras', 'missile', 'immigration',
               'synfuels-corporation-cutback', 'education', 'superfund-right-to-sue',
               'crime', 'duty-free-exports', 'exportafrica']


df = pd.read_csv(url, names=columnsName, sep=',')
# df['exportafrica'] = df['exportafrica'].str.replace('?', 'n')
print(df.columns)
print(df.head())
for colnames in df.columns:
    df[colnames] = df[colnames].str.replace('?', 'n')

print(df.head())
print(df['exportafrica'].head())


# sns.countplot(x='education', hue='party', data=df, palette='RdBu')
# plt.xticks([0, 1], ['No', 'Yes'])
# plt.pause(1)
# plt.clf()


# sns.countplot(x='missile', hue='party', data=df, palette='RdBu')
# plt.xticks([0, 1], ['No', 'Yes'])
# plt.pause(1)
# plt.clf()


# sns.countplot(x='satellite', hue='party', data=df, palette='RdBu')
# plt.xticks([0, 1], ['No', 'Yes'])
# # plt.pause(3)
# # plt.clf()
# plt.show()

 


# digits = datasets.load_digits()

# # Print the keys and DESCR of the dataset
# print(digits.keys())
# print(digits.DESCR)

# # Print the shape of the images and data keys
# print(digits.images.shape)
# print(digits.data.shape)

# # Display digit 1010
# plt.imshow(digits.images[1010], cmap=plt.cm.gray_r, interpolation='nearest')
# plt.show()


y = df['party'].values
X = df.drop('party', axis=1).values


# Split into training and test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)

    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()
