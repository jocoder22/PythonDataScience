#!/usr/bin/env python
import os
import re
import PyPDF2


path2 = r'C:\Users\Jose\Desktop\PythonDataScience\datavisualization'
os.chdir(path2)

def readpdf(regex):
    with open('matplotlibrc_copy2.pdf', 'rb') as mplt:
        listc = []
        reader = PyPDF2.PdfFileReader(mplt)
        for page in range(reader.numPages):
            page_n = reader.getPage(page)
            text = page_n.extractText()
            result = re.findall(regex, text)
            if len(result): print(result)


friends = '''John has 4 sisters while Peter loaded 5 disc and Mary boiled 3 chickens
            for Mummy 40 years celebrate working for Perking 20 months, 9987 66
            we sang 4t4t8te 7t5g6 in the mama gaga pump hall in the 4u8k9b0 afternoon'''

num = 'This is just repeated 2289 8798 and 667 '
words = r'([A-Za-z]+)\s(\d+)\s(\w+)'
numbers = r'\d+\s\d+\s'
readpdf(words)
print(re.findall(words, friends))

# Repeatations

print(re.findall(r"\s(\d[A-Za-z])\1", friends))
print(re.findall(r'\s(\d{2,})\s', num))
print(re.search(r'\w*(\d)\1\w*', num).group(0))
print(re.findall(r'(\d+)', num))
print(re.findall(r'(\d)+', num))

