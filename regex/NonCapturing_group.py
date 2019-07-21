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
regex = re.compile(r'(?:[a-z]\d){2,}')
repeated = regex.findall(luckynumbers)
print(repeated)