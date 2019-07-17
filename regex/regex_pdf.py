#!/usr/bin/env python
import os
import re
import PyPDF2
from collections import defaultdict

path2 = r'C:\Users\Jose\Desktop\PythonDataScience\datavisualization'
os.chdir(path2)
with open('matplotlibrc_copy2.pdf', 'rb') as mplt:
    listc = []
    reader = PyPDF2.PdfFileReader(mplt)
    d = defaultdict(int)
    for page in range(reader.numPages):
        page_n = reader.getPage(page)
        text = page_n.extractText()
        listc.extend(re.findall(r'lines', text))
        for word in re.split(" |#|\n|$|'|,", text):
            d[word] += 1
        

    print(len(listc))

    print(d.items())

