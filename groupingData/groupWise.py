import os
import numpy as np
import pandas as pd


def print2(*args):
    for arg in args:
        print(arg, end='\n\n')
        
sp = {"sep":"\n\n", "end":"\n\n"} 
  
      
os.chdir("C:/Users/Jose/Documents/PythonDataScience1/Code/Code/Section 3")
irisdata = pd.read_csv("iris.csv")

# Explore dataset
irisdata.head()
irisdata.tail()
irisdata.shape
irisdata.columns

# replace missing
fillingdata = np.array([False] * (150 * 5))
fillingdata[np.random.choice(np.array(150 * 5), size=150, replace=False)] = True
fillingdata = fillingdata.reshape(150, 5)
fillingdata[:, 4] = False
fillingdata[:10, :]

# convert to dataFrame
pdata = pd.DataFrame(fillingdata, index=irisdata.index, columns=irisdata.columns)
pdata.head()

# apply groupwise transformation
iriscopy = irisdata.copy()
iriscopy[fillingdata] = np.nan
iriscopy.head()

iriscopy2 = irisdata.copy()
iriscopy2[pdata] = np.nan
iriscopy2.head()

# group the dataset according to species
irisgroup = iriscopy.groupby('species')
irisgroup.head()
irisgroup.groups
iriscopy.mean()

# generate group means for each column
gmean = lambda x: x.fillna(x.mean())
irisfillna = irisgroup.apply(gmean)
irisfillna.head()
irisfillna.tail()



# apply standardization to entire columns with regard to group
# first split the dataset
stdgroup = irisdata.groupby('species')
stdgroup.groups

# generate lambda function
gstandard = lambda x: (x - x.mean()) / x.std()
irisStd = irisfillna.loc[:, 'sepal_length':'petal_width'].apply(gstandard)
irisStd.head()

# subset dataset
subsetdata = stdgroup[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
irisStd2 = subsetdata.apply(gstandard)
stdgroup.head()


# Remove the index
iStd2 = irisStd.reset_index()
irisjoin = irisdata.join(iStd2, rsuffix='_std')
irisjoin.head()


# Join the dataset
irisStd2 = stdgroup[['sepal_length', 'sepal_width', 'petal_length', 
                       'petal_width']].apply(gstandard)
irisdata = irisdata.join(irisStd2, rsuffix='_Standandized')




# apply standardization to entire columns without regard to group
# first drop species
drop_species = irisdata.drop(columns='species')
# drop_species = irisdata.drop('species', axis=1)


# apply standardization
iris_stand = drop_species.apply(gstandard)
iris_stand.head()
iris_stand.tail()