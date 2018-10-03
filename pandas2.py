import numpy as np
import pandas as pd
from pandas import Series, DataFrame

series1 = Series([12, 23, 45, 21, 27, 31, 33])
series2 = Series(['Adam', 'Eve', 'kelly', 'Ben', 'Mary', 'John'])

print(series1)
print(series2)
type(series1)  # <class 'pandas.core.series.Series'>


# creating index;
indx = pd.Index(["USA", "Canada", "Algeria",
                 "Mexico", "Japan", "Kenya",
                 "Malaysia", "Holland", "Poland",
                 "Brazil", "South Korea", "China"])

print(indx)
type(indx)  # <class 'pandas.core.indexes.base.Index'>

Popl = Series([8902, 4893, 560,
               849, 510, 290,
               486, 409, np.nan,
               569, 954, 9840],
              index=indx, name="Population")
print(Popl)

# creating series with dictionary;
Sales = Series({"Tv": 459.89, "Radio": 250.98, "Laptop": 1245.99,
                "Telephone": 57.99, "Ipad": 810.98, "Washer": 2690},
               name="Electronics")
print(Sales)

Countries = Series({"USA": "Washington DC", "Japan": "Tokyo",
                    "South Korea": "Seoul", "Algeria": "Algiers",
                    "Brazil": "Brasilia", "China": "Beijing",
                    "Kenya": "Nairobi"}, name="countries")

print(Countries)


# Creating DataFrames;
# from numpy array;
nmdata = np.arange(6, 18).reshape(3, 4)
mmdata = np.arange(0, 9).reshape(3, 3)
print(mmdata)
print(nmdata)

# Form DataFrame;
print(DataFrame(nmdata))


# Adding DataFrame labels;
print(DataFrame(nmdata,
                index=["Book", "Radio", "Heater"],
                columns=["Price", "Discount", "Membership", "Sales"]))
