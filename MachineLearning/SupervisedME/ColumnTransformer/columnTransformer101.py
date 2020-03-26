import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error 

from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import FunctionTransformer

from sklearn.preprocessing import OneHotEncoder
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier


from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor, VotingRegressor, RandomForestRegressor 
from sklearn.linear_model import LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor, RidgeCV, LassoCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor 

from xgboost import XGBRegressor

# sns.set(style="ticks", color_codes=True)
plt.rc('figure', figsize=[12,8], dpi=100)


# custom transformer
class columndropper(BaseEstimator, TransformerMixin):
    """columnSelector class select columns"""
    # def __init__(self, col=None):
        
    #     self.columnlist = col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        
        col_ = X.drop()
        
        return col_


f_dummy = FunctionTransformer(lambda x : pd.get_dummies(x, columns=['neighbourhood_group', 'room_type']), validate=False)

sp = {"end":'\n\n', "sep":'\n\n'}

airbnb = "http://data.insideairbnb.com/united-states/ny/new-york-city/2020-03-13/visualisations/listings.csv"
df = pd.read_csv(airbnb)

select_col = ['neighbourhood_group',
        'latitude', 'longitude', 'room_type', 'price',
       'minimum_nights', 'number_of_reviews', 
       'reviews_per_month', 'calculated_host_listings_count',
       'availability_365']

dups = ['neighbourhood_group',
        'latitude', 'longitude', 'room_type', 
       'minimum_nights', 'number_of_reviews', 
       'reviews_per_month', 'calculated_host_listings_count']



df.drop_duplicates(subset = dups, keep = False, inplace = True)
data = df[select_col]


print(data['neighbourhood_group'].value_counts().index.tolist())
data["strata"] = data['neighbourhood_group'].map(
    {
       'Manhattan': "M",
       'Brooklyn':"B", 
       'Queens':"Q", 
       'Bronx':"B", 
       'Staten Island':"S" 
    }
)
data[['neighbourhood_group', 'room_type']] = data[['neighbourhood_group', 'room_type']].astype("category")
# data = pd.get_dummies(data, columns=['neighbourhood_group', 'room_type'])

data.dropna(inplace=True)

data2 = data.copy()

target = data.pop("price")
data['st2'] = data[['strata']]
data.drop(columns=["strata"], inplace=True)

print(data.head(), data.info(), **sp)

x_train, x_test, y_train, y_test = train_test_split(data.iloc[:,:-1], target, test_size=0.2, stratify=data['st2'])

num = data.select_dtypes(include=['int64', 'float64']).columns

print(x_train.head(), **sp)

# the sequence
# 1. dummy cat
# du = ("dummy", f_dummy, " ")
du = ("dummy", OneHotEncoder(), [0, 3])

# 2. StandardScaler scale
scl = ("scaler", StandardScaler(), num)

transformer_list = [du, scl]
transformer = ColumnTransformer(transformers=transformer_list)

model = RandomForestRegressor(n_estimators = 200, n_jobs = -1,
                           oob_score = True, bootstrap = True)

pl = Pipeline([
                ('prep', transformer),
                ('model', model)
                     ])

pl.fit(x_train, y_train)
pred = pl.predict(x_test)


print("mean_squared_error: ", mean_squared_error(y_test, pred))
print("Root mean_squared_error: ", np.sqrt(mean_squared_error(y_test, pred)))
print("R_squared : ", pl.score(x_test, y_test))
print("R2_squared : ", r2_score(y_test, pred))




estimatorstack = [
    ('Random Forest', RandomForestRegressor(random_state=342)),
    ('Lasso', LassoCV())
        ]


stacking_regressor = StackingRegressor(
    estimators=estimatorstack, final_estimator=RidgeCV())


estimators = [('OLS', LinearRegression()),
              ('Theil-Sen', TheilSenRegressor(random_state=682)),
              ('RANSAC', RANSACRegressor(random_state=9052)),
              ('HuberRegressor', HuberRegressor()),
              ("decisionTree", DecisionTreeRegressor()),
              ("radomForest", RandomForestRegressor(n_estimators = 200,
                           n_jobs = -1,
                           oob_score = True,
                           bootstrap = True,
                           random_state = 3452)), 
            ('Stacked Regressors', stacking_regressor),
            ("MLregs", MLPRegressor(hidden_layer_sizes=(100,100),
                                    tol=1e-2, max_iter=5000, random_state=670))
            ]

for name, estimator in estimators:
    model = make_pipeline(transformer, estimator)
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    print(name," mean_absolute_error : ", mean_absolute_error(y_test, pred))
    print(name," mean_squared_error: ", mean_squared_error(y_test, pred))
    print(name," Root mean_squared_error: ", np.sqrt(mean_squared_error(y_test, pred)))
    print(name," R2_squared : ", r2_score(y_test, pred))
    print(name," R_squared : ", model.score(x_test, y_test), end="\n\n")






target =  pd.get_dummies(data2[['neighbourhood_group']])
data2['st2'] = data2['strata']

data2.drop(columns=["strata", "neighbourhood_group"], inplace=True)
num = data2.select_dtypes(include=['int64', 'float64']).columns


x_train, x_test, y_train, y_test = train_test_split(data2.iloc[:,:-1], target, test_size=0.2, stratify=data2['st2'])


# the sequence
# 21 dummy cat
# du = ("dummy", f_dummy, " ")
du = ("dummy", OneHotEncoder(), ["room_type"])

# 2. StandardScaler scale
scl = ("scaler", StandardScaler(), num)

transformer_list = [du, scl]
transformer = ColumnTransformer(transformers=transformer_list)

model = MultiOutputClassifier(RandomForestClassifier())

pl2 = Pipeline([
                ('preprocess', transformer),
                ('model', model)
                     ])

pl2.fit(x_train, y_train)
pred = pl2.predict(x_test)


# # Calculate accuracy
accuracy = (pred == y_test).mean().mean()
accuracyscore = pl2.score(x_test, y_test)


print(f"Model Accuracy: {accuracy*100:.02f}%\n")
print(f"Model Accuracy: {accuracyscore*100:.02f}%\n")
