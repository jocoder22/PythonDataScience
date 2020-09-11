import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from printdescribe import print2

path = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"

data = pd.read_excel(path,header=1, index_col=0)
data = data.rename(columns={'default payment next month':"default"})

X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-1], data.iloc[:,-1], stratify=data.iloc[:,-1], test_size=0.3)
print2(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# Set the learning rates & results storage
learning_rates = [0.001, 0.01, 0.05,0.08, 0.1, 0.2 , 0.3, 0.5]
results_list = []

# Create the for loop to evaluate model predictions for each learning rate
for lr in learning_rates:
    model = GradientBoostingClassifier(learning_rate=lr)
    predictions = model.fit(X_train, y_train).predict(X_test)
    # Save the learning rate and accuracy score
    results_list.append([lr, accuracy_score(y_test, predictions)])

# Gather everything into a DataFrame
results_df = pd.DataFrame(results_list, columns=['learning_rate', 'accuracy'])
print2(results_df)


target_names = ['class 0', 'class 1']
print2(classification_report(y_test, predictions, target_names=target_names))

# import os
# print(os.cpu_count())

# from sklearn import metrics
# print2(metrics.SCORERS.keys())

# Create a Random Forest Classifier with specified criterion
rfclass = RandomForestClassifier(criterion='entropy', n_estimators=100)

# Create the parameter grid
param_grid = {'max_depth': [2, 4, 8, 15], 'max_features': ['auto', 'sqrt']} 

# Create a GridSearchCV object
grid_rfclass = GridSearchCV(
    estimator=rf_class,
    param_grid=param_grid,
    scoring='roc_auc',
    n_jobs=-1,
    cv=5,
    refit=True, return_train_score=True)
print(grid_rfclass)


predictions = grid_rfclass.fit(X_train, y_train).predict(X_test)
results = pd.DataFrame(grid_rfclass.cv_results_)
print(results.shape)

print2(results.iloc[:,:9], results.iloc[:,9:18], results.iloc[:,18:])


print2(results[results["rank_test_score"]==1])
print2(grid_rfclass.best_estimator_, grid_rfclass.best_index_, grid_rfclass.best_params_, grid_rfclass.best_score_)


pd.set_option("display.max_colwidth", -1)
print(results.loc[:,"params"])


# Create a variable from the row related to the best-performing square
best_row = results.loc[[grid_rfclass.best_index_]]
print(best_row)

# Get the n_estimators parameter from the best-performing square and print
best_n_estimators = grid_rfclass.best_params_["max_depth"]
print(best_n_estimators)


# See what type of object the best_estimator_ property is
print(type(grid_rfclass.best_estimator_))

# Create an array of predictions directly using the best_estimator_ property
predictions = grid_rfclass.best_estimator_.predict(X_test)

# Take a look to confirm it worked, this should be an array of 1's and 0's
print(predictions[0:5])

# Now create a confusion matrix 
print("Confusion Matrix \n", confusion_matrix(y_test, predictions))

# Get the ROC-AUC score
predictions_proba = grid_rfclass.best_estimator_.predict_proba(X_test)[:,1]
print("ROC-AUC Score \n", roc_auc_score(y_test, predictions_proba))




import random
# Create lists for criterion and max_features
criterion_list = ['gini', 'entropy']
max_feature_list = ["auto", "sqrt", "log2", "None"]

learning = np.linspace(0.001,2,150)

# Create a list of values for the max_depth hyperparameter
max_depth_list = list(range(3,56))

# Combination list
combinations_list = [list(x) for x in product(max_depth_list, learning, max_feature_list)]

# Sample hyperparameter combinations for a random search
combinations_random_chosen = random.sample(combinations_list, 150)

# Print the result
print(combinations_random_chosen)



nn = pd.DataFrame(combinations_random_chosen, columns = ["max_depth", "learning_rate", 'max_feature'])
# plt.scatter(nn.max_feature, nn.max_depth, cmap=nn.criterion)
# plt.gca().set(ylabel = "Min Sample leaf", xlabel="Learing Rate")
# plt.show()

groups = nn.groupby('max_feature')
fig, ax = plt.subplots()
# ax.set_color_cycle(colors)
ax.margins(0.05)
for name, group in groups:
    ax.plot(group.learning_rate, group.max_depth, marker='o', linestyle='', ms=12, label=name)
ax.legend(numpoints=1, loc='upper left')

plt.show()


nn['classes'] = pd.Categorical(nn['max_feature']).codes
pp3 = nn.max_feature.value_counts()
pp4 = tuple(pp3.index)


pp = np.unique(nn.classes).tolist()
pp2 = tuple(pp)

plt.scatter(nn.max_depth, nn.learning_rate, alpha=0.5,
            s=nn.max_depth*2, c=nn.classes, cmap='viridis')
plt.gca().set(xlabel=nn.columns[0], ylabel=nn.columns[1])
plt.legend(tuple(pp), tuple(pp3.index), loc='upper left');


import seaborn as sns
sns.pairplot(x_vars=["max_depth"], y_vars=["learning_rate"], data=nn, hue="max_feature", size=10);
