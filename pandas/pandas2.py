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

# From series;
ssss = DataFrame({"Pool": series1, "Legg": series2})
ssss = DataFrame({"Pool": series1, "Legg": series2}).T
ssss = DataFrame([series1, series2])

# Form DataFrame;
print(DataFrame(nmdata))


# Adding DataFrame labels;
print(DataFrame(nmdata,
                index=["Book", "Radio", "Heater"],
                columns=["Price", "Discount", "Membership", "Sales"]))


# creating dataframe from tuples;
tuple1 = [(25, 'Boy'), (18, "Girl"),
          (46, 'Man'), (35, 'Woman')]

print(tuple1)
print(DataFrame(tuple1, columns=["Age", "Role"]))


# create dataframe from dictionary;
print(DataFrame({"Prices": [234, 689, 157],
                 "Product": ["Pen", "Pencils", "Crayons"]}))

# Arrays must all be same length
print(DataFrame({"Prices": [5234, 7689, 9157],
                 "Product": ["Books", "Crayons"]}))  # produce error


# Using Dictionary and Series, DataFrame produce no error
# This is because the series are already indexed  ;
# NaN is used to filled for missing values;
print(DataFrame({"Names": series2, "Age": series1}))


# form DataFrame from dictionary;
print(DataFrame([Popl, Countries, Sales]))  # Not really what we want

print(DataFrame({"Population": Popl,
                 "Countries": Countries,
                 "Electronics": Sales}))   # this is it!


# or we can use the transpose function .T
print(DataFrame([Popl, Countries, Sales]).T)  # works fines


# using the append and concat function;
Popl.append(Series({"Uganda": 2345, "Ghana": 2309}))  # returns a new series
print(Popl)  # return the old series, new series not permanently added

dpnd = (DataFrame([Popl, Countries, Sales]).T)
dpnd.append(DataFrame({"Population": Series({"Uganda": 1389, "Ghana": 1209}),
                       "countries": Series({"Uganda": "Kampala",
                                            "Ghana": "Accra"}),
                       "Electronics": Series({"Uganda": np.nan,
                                              "Ghana": np.nan})}))


# using concat function; This is long and complicated;
pd.concat([dpnd, DataFrame({"numbers": Series(np.arange(12),
                                              index=Popl.index),
                            "Letters": Series(["aiu", "ag", "hf", "tr",
                                               "bsd", "srsgrf", "adf",
                                               "opf", "uty", "dgd",
                                               "dgd", "oert"],
                                              index=Popl.index)})],
          axis=1)


# Saving the data:
dpnd.to_csv("C:/Users/.../.../mydata2.csv")
dpnd.to_csv("C:/Users/.../.../mydata2.txt")
dpnd.to_json("C:/Users/.../.../mydata2.json")
dpnd.to_html("C:/Users/.../.../mydata2.html")
dpnd.to_pickle("C:/Users/.../.../mydata2.npy")
