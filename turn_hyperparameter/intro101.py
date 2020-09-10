import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split as tts
from sklearn import tree

import matplotlib.pyplot as plt

from printdescribe import print2, describe2

# please install graphviz and pydotplus

path = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
data = pd.read_excel(path, header=1, index_col=0)

print2(data.head(), data.info(), data.shape)

data["default payment next month"].value_counts()

data = data.rename(columns={"default payment next month" : "default"})
print2(data.head())
prop = data["default"].value_counts(normalize=True)*100
print2(prop)


X_train, X_test, y_train, y_test = tts(data.iloc[:,:-1], data.iloc[:,-1], test_size=0.3, 
                                       stratify=data.iloc[:,-1], random_state=42)

print2(X_train.shape)

model_lg = LogisticRegression(solver="lbfgs", max_iter=10000)
model_lg.fit(X_train, y_train)
print2(model_lg.coef_)


# var_names = list(data.iloc[:,:-1].columns)
var_names = list(X_train.columns)
zipper = list(zip(var_names, model_lg.coef_[0]))
print2(zipper)

coeffs = [list(x) for x in zipper]
print2(coeffs)

coeffs = pd.DataFrame(coeffs, columns=["Variable", "Coefficient"])
coeffs.sort_values(by=["Coefficient"], axis=0, inplace=True, ascending=False)
print2(coeffs.head(), coeffs.tail())



val_names2 = X_train.columns.tolist()
model_coeffs = model_lg.coef_[0]
coeff_data = pd.DataFrame({"Variable":val_names2, "Coefficient":model_coeffs})
sorted_coeffs = coeff_data.sort_values(by=["Coefficient"], axis=0, ascending=False)[0:5]
print2(sorted_coeffs)


model_rfc = RandomForestClassifier(n_estimators=100, max_depth=2)
print2(model_rfc)

model_rfc.fit(X_train, y_train)

chosen_tree = model_rfc.estimators_[7]
# visualize the left, second form top node
# get the column it split on
split_col = chosen_tree.tree_.feature[1]
split_col_name = X_train.columns[split_col]


# get the level it split on
split_value = chosen_tree.tree_.threshold[1]
print2(f"This is the node split on feature {split_col_name}, at a value of {split_value}")


plt.rcParams["figure.figsize"] = 21,16
imgplot = tree.plot_tree(chosen_tree)
plt.show()


fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (2,2), dpi=400)
feature_name = list(data.iloc[:,:-1].columns)
target_name = list(data.iloc[:, [-1]].columns)
tree.plot_tree(chosen_tree,
               feature_names = feature_name, 
               class_names=target_name,
               filled = True);
# fig.savefig('choosen_7.png')
plt.show()
