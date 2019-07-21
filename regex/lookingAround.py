#!/usr/bin/env python
import re
import os
from collections import defaultdict

path = r"C:\Users\Jose\Desktop\PythonDataScience\regex"
os.chdir(path)

def printt(*args):
    for a in args:
        print(a, end='\n\n')

# Searching file without implicit opening
with open('lorem.txt', 'r') as doc:
    text = doc.read()
    printt(re.findall(r'\d+', text))