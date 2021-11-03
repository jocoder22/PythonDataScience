import os
import numpy as np
import pandas as pd


# empty dictionary

dic1 = {}

# filled dictionary

dict2 = {"boll": [4, 5, 6, 7, 90], "gii": [9, 4, 12, 34, 74], "kut": [8, 93, 61, 43, 90]}

# dictionary keys 
print(dict2.keys())


# dictionary values
print(dict2.values())


# dictionary key: value pairs
for item in dict2.items():
    print(item)

# get the keys and values 
for item in dict2.items():
    print(f"keys: {item[0]},  The values: {item[1]}")


# Add value to dictionary
dict2["gii"].append(984)

# Add new key, value
dict2["hoo"] = [948]

# print the sorted keys
print(sorted(list(dict2)))

# better way to get the keys and values
for key, value in dict2.items():
    print(f"keys: {key},  The values: {value}")


# dictionary comprehension
dict3 = {f"key{i}":v*5 for i,v in enumerate(reversed(range(0,6)))}
print(dict3)

# nested dictionary
nested = {f"dictionary{d1}": {d2: d1 ** d2 for d2 in range(1, 8)} for d1 in range(1, 5)}

print(nested)

# dataframe from dictionary
dict22 = {"boll": [4, 5, 6, 7, 90], "gii": [9, 4, 12, 34, 74], "kut": [8, 93, 61, 43, 90]}

# Using keys as column name, orient = None
df = pd.DataFrame.from_dict(dict22, orient="columns")
print(df)

# using keys as index
df = pd.DataFrame.from_dict(dict22, orient="index",
                                columns=[f"col{i}" for i in range(1,6)])
print(df)

# delete and clear
# del dict2["gii"] #  => delete the entry gii
# del dict2    # = > delete the whole dictionary
# dict2.clear() # => clear all entry in the dictionary, returns empty dictionary
print(dict2)