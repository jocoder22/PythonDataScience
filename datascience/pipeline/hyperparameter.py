from sklearn.datasets import load_digits
from sklearn import svm
from sklearn import model_selection
from sklearn.model_selection import train_test_split

digits = load_digits()
print(digits.DESCR)
ddata, dtarget = digits.data,  digits.target
ddata.shape

# set up support vector machine algorithm
hs = svm.SVC()
hprob = svm.SVC(probability=True, random_state=1)


# set the search grid for model selection
search_grid = [
        {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
        {'C': [1, 10, 100, 1000], 'gamma': [0.01, 0.001, 0.0001], 
        'kernel': ['rbf']},
        ]
scorer = 'accuracy'

# set up the model searh function
searchFunc = model_selection.GridSearchCV(
                estimator=hs,  param_grid=search_grid,
                scoring=scorer,  n_jobs=-1, iid=False,
                refit=True, cv=10)