import numpy as np 
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split as splitit
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

irisdata = datasets.load_iris()
features_ = irisdata.data[:, [2, 3]]
target_ = irisdata.target

print(np.unique(target_))


# Using the full dataset
features_2 = irisdata.data
# Standardization
stardard = StandardScaler()
# Standardization with the original data
stardard.fit(features_2)
features_std = stardard.transform(features_2)
print(features_std[:5, :])

# Split features into test and training datasets
X_train, X_test, y_train, y_test = splitit(features_std, target_, test_size=0.3,
                                       random_state=1, stratify=target_)


# fit the model
model_2 = Perceptron(n_iter=150, eta0=0.01, random_state=1)
model_2.fit(X_train, y_train)
y_pred = model_2.predict(X_test)


print('Accuracy: {:.3f}'.format(accuracy_score(y_test, y_pred)))
print('Accuracy: {:.2f}%'.format(model_2.score(X_test, y_test) *100))
# Accuracy: 95.56%



# Using only 2 features
# split the dataset
X_train, X_test, y_train, y_test = splitit(features_, target_, test_size=0.3,
                                       random_state=1, stratify=target_)

print('Labels counts in target:', np.bincount(target_))
print('Labels counts in y_train:', np.bincount(y_train))
print('Labels counts in y_test:', np.bincount(y_test))

# Standardization
stardard = StandardScaler()
# Standardization with the original data
stardard.fit(features_)
X_train_std = stardard.transform(X_train)
X_test_std = stardard.transform(X_test)

# fit the model
model_1 = Perceptron(n_iter=40, eta0=0.01, random_state=1)
model_1.fit(X_train_std, y_train)
y_pred = model_1.predict(X_test_std)


print('Accuracy: {:.3f}'.format(accuracy_score(y_test, y_pred)))
print('Accuracy: {:.2f}%'.format(model_1.score(X_test_std, y_test) *100))
# Accuracy: 93.33%