import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def print2(*args):
    for arg in args:
        print(arg, sep="\n\n", end="\n\n")

path = "D:\PythonDataScience\datavisualization\Uda_datavisualization"
os.chdir(path)

fcon = pd.read_csv('fuel-econ.csv')

url = "http://data.insideairbnb.com/united-states/ny/new-york-city/2020-03-13/visualisations/listings.csv"

airbnb = pd.read_csv(url)

print2(airbnb.head(), airbnb.columns, airbnb.info())

# bivariate both numeric
plt.scatter(data=airbnb, x="latitude", y="longitude")
plt.show()

plt.scatter(data=airbnb, x="price", y="longitude")
plt.show()


airbnb2 = airbnb.loc[airbnb.price < 200]
airbnb2['logprice'] = np.log10(airbnb2['price'])
sns.regplot(data=airbnb2, x="logprice", y="number_of_reviews")
plt.show()