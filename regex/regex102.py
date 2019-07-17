#!/usr/bin/env python
import re
import os

path = r"C:\Users\Jose\Desktop\PythonDataScience\regex"
os.chdir(path)

def printt(*args):
    for a in args:
        print(a, end='\n\n')


lorem = """Lorem Ipsum is simply dummy text of the printing and typesetting industry. 
        Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, 
        when an unknown printer took a galley of type and scrambled it to make a type 
        specimen book. It has survived not only five centuries, but also the leap into. 
        Loren electronic typesetting, remaining essentially unchanged. It was popularised in 
        the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, 
        and more recently with desktop publishing software like Aldus PageMaker including 
        versions of Lorem Ipsum"""

# locational Metacharacters: ^ or \A and $ or \Z, at the begining and the end of  whole string
beginword = re.findall(r'^L\w+', lorem)
endword = re.findall(r'I\w+$', lorem)
printt(beginword, endword)

# \b or opposite (\B) finds within the whole string
endsum = re.findall(r'\w+ing\b', lorem)
startwith = re.findall(r'\bun\w+', lorem)
startwith2 = re.findall(r'\AL\w+', lorem)
numbers = re.findall(r'\d+', lorem)
printt(endsum, startwith, startwith2, numbers)


# Searching file without implicit opening
with open('lorem.txt', 'r') as text:
    print(re.findall(r'\d+', text.read()))

path2 = r'C:\Users\Jose\Desktop\PythonDataScience\datavisualization'
os.chdir(path2)
with open('matplotlibrc_copy.txt', 'r') as mplt:
    print(re.findall(r'\d+', mplt.read()))c