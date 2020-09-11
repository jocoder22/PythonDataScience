import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
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
