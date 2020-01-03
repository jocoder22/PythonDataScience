import re
import pandas as pd
import numpy as np 
from numpy import NaN

def print2(*args):
    for arg in args:
        print(arg, end='\n\n')
  


tips = pd.read_csv('tips.csv')
df = pd.read_csv('dob_job_application_filings_subset.csv')
df_subset = df[:, 2:]

# Convert the sex column to type 'category'
tips.sex = tips.sex.astype('category')

# Convert the smoker column to type 'category'
tips.smoker = tips.smoker.astype('category')

# Print the info of tips
print(tips.info())



# Convert 'total_bill' to a numeric dtype
# errors = coerce will force any other forms to NaN
tips['total_bill'] = pd.to_numeric(tips['total_bill'], errors='coerce')

# Convert 'tip' to a numeric dtype
tips['tip'] = pd.to_numeric(tips['tip'], errors='coerce')

# Print the info of tips
print(tips.info())





# Using the regular expression module, to recode values
# Compile the pattern: prog
prog = re.compile(r'\d{3}-\d{3}-\d{4}')

# See if the pattern matches
result = prog.match('123-456-7890')
print(bool(result))

# See if the pattern matches
result2 = prog.match('1123-456-7890')
print(bool(result2))


# When using a regular expression to extract multiple numbers
#  (or multiple pattern matches, to be exact), you can use the 
#  re.findall() function. You pass in a pattern and a string to re.findall(), 
#  and it will return a list of the matches.

# Find the numeric values: matches
matches = re.findall(r'\d+', 'the recipe calls for 10 strawberries and 1 banana')

# Print the matches
print(matches)



# Write the first pattern
pattern1 = bool(re.match(pattern=r'\d{3}-\d{3}-\d{4}', string='123-456-7890'))
print(pattern1)

# Write the second pattern
pattern2 = bool(re.match(pattern=r'\$\d*\.\d{2}', string='$123.45'))
print(pattern2)

# Write the third pattern
pattern3 = bool(re.match(pattern=r'[A-Z]\w*', string='Australia'))
print(pattern3)





# write a data cleaning function
# Define recode_gender()
def recode_gender(gender):

    # Return 0 if gender is 'Female'
    if gender == 'Female':
        return 0
    
    # Return 1 if gender is 'Male'    
    elif gender == 'Male':
        return 1
    
    # Return np.nan    
    else:
        return np.nan



# Apply the function to the sex column
tips['recode'] = tips.sex.apply(recode_gender)

# Print the first five rows of tips
print(tips.head())



# this is row wise application of apply function
# default is axis=0 for column wise application

pattern = r'^\$\d*\.\d{2}$'

def diff_money(row, pattern):
        icost = row['Initial Cost']
        tef = row['Total est. fee']

        if bool(pattern.match(icost)) and bool(pattern.match(tef)):
                icost = icost.replace("$", "") 
                tef = tef.replace("$", "")

                icost = float(icost)
                tef = float(tef)

                return icost - tef
        
        else:
                return (NaN)


df_subset['dfff'] = df_subset.apply(diff_money, axis=1, pattern=pattern)






# Write the lambda function using replace
tips['total_dollar_replace'] = tips.total_dollar.apply(lambda x: x.replace ('$', ''))

# Write the lambda function using regular expressions
tips['total_dollar_re'] = tips.total_dollar.apply(lambda x: re.findall(r'\d+\.\d+', x)[0])

# Print the head of tips
print(tips.head())








