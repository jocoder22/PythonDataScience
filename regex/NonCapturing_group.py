#!/usr/bin/env python
import re
import os
from collections import defaultdict

path = r"C:\Users\Jose\Desktop\PythonDataScience\regex"
os.chdir(path)

def printt(*args):
    for a in args:
        print(a, end='\n\n')


luckynumbers = '2345, 7e7e56, 88e24t, 44r9h8, 86n9k6, 3389, 5611 and 5509'
num2 = 'There are too much 5578 and 33'


# Non capturing group
### if only one group, return the non-capturing group
### in this case, return the match which is the group zero
regex = re.compile(r'(?:[a-z]\d){2,}')
regex2 = re.compile(r'(\d)\1')
repeated = regex.findall(luckynumbers)
consecutive_num = regex2.findall(luckynumbers)
printt(repeated, consecutive_num)

mywords = 'It is priced 200 dollars, soled for 220 euros or mintes'
print(re.findall(r'\d+\s(?:dollars|euros|mintes)', mywords))
print(re.findall(r'(\d+)\s(dollars|euros|mintes)', mywords))

### if more than 1 group, the non-capturing group has no number
### the non-capturing group is not captured and therefore not returned
sales = ''' Books 100 $200 is the total cost of books
            CPU 30 $30000 total delayed payment
            Lamps 10 $400 found in stores
            Acs 3 $2010 awaiting returns
            Beds 10 $8000 for clearance'''

items = re.compile(r'(\w+)(?:\s\d+\s)(\$\d+)')
print(items.findall(sales))

