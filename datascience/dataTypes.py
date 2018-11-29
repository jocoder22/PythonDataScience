import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import fetch_20newsgroups

categoricalFeature = pd.Series(['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                                'Friday', 'Saturday', 'Sunday'])

catmapping = pd.get_dummies(categoricalFeature)

catmapping['Friday']
catmapping['Tuesday']


# Using sklearn
label1 = LabelEncoder()
hot1 = OneHotEncoder()
weekdays = ['Monday', 'Tuesday', 'Wednesday',
            'Thursday', 'Friday', 'Saturday', 'Sunday']
fit_data = label1.fit_transform(weekdays)
hot1.fit([[fit_data[0]], [fit_data[1]], [fit_data[2]],
          [fit_data[3]], [fit_data[4]]])

print(hot1.transform([label1.transform(['Monday'])]).toarray())
print(hot1.transform([label1.transform(['Tuesday'])]).toarray())



# Working with text
categories = ['sci.med', 'sci.space']
twenty_sci_news = fetch_20newsgroups(categories=categories)