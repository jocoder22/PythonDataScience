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
            for Mummy 40 years celebrate working for Perking 20 months
            we sang tetate t5g6 in the mama gaga pump hall in the afternoon'''
words = r'([A-Za-z]+)\s(\d+)\s(\w+)'
numbers = r'\d+\s\d+\s'
readpdf(words)
print(re.findall(words, friends))

# Repeatations


