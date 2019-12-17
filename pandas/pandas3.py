#!/usr/bin/env python
import numpy as np
import pandas as pd
from datetime import date
from datetime import timedelta
from string import ascii_letters, digits, ascii_lowercase, ascii_uppercase

def print2(*args):
    for arg in args:
        print(arg, end='\n\n')
  
sp = {"sep":"\n\n", "end":"\n\n"} 

users = 'Bob John Peter James Anna Mary'.split()
password = [''.join(np.random.choice(list(ascii_letters + digits), 12)) for x in range(len(users))]
listDate = [date.today() - timedelta(days=int(x)) for x in np.random.randint(2, 100, len(users))]
FirstName = ["".join(np.random.choice(list(ascii_lowercase), 10)).capitalize() for x in range(len(users))]
LastName = ["".join(np.random.choice(list(ascii_lowercase), 8)).capitalize() for x in range(len(users))]
email = [''.join(np.random.choice(list(ascii_letters + digits), 8)) + '@gmail.com' for x in range(len(users))]

df1 = pd.DataFrame({'DateEnroll':pd.to_datetime(listDate), 'User':users, 'FirstName':FirstName, 'LastName':LastName,
                     'Email':email, 'Password':password})
df1.loc[6] = [pd.to_datetime(date(2019,11,22)), "Jane", "Vaictme", "Moandad", "jand_844@gmail.com", "iedkadlduy58"]

df1.drop(2, inplace=True)
df1.drop('Password', axis=1, inplace=True)

myindex = ["".join(np.random.choice(list(ascii_uppercase), 3)) + "00" + str(x) for x in range(len(users))]
df1.index = myindex
df1.index.name = "RegNum"

print2(df1, df1.columns, df1.index)

df1.reset_index(inplace=True)
print2(df1, df1.columns, df1.index)

df1.set_index(['RegNum', 'User'], inplace=True)
print2(df1, df1.columns, df1.index)