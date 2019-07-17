#!/usr/bin/env python
import os
import re
import PyPDF2

path2 = r'C:\Users\Jose\Desktop\PythonDataScience\datavisualization'
os.chdir(path2)
with open('matplotlibrc_copy2.pdf', 'rb') as mplt:
    listc = []
    reader = PyPDF2.PdfFileReader(mplt)
    for pages in range(reader.numPages):
        page_n = reader.getPage(pages)
        listc.extend(re.findall(r'lines', page_n.extractText()))
    print(len(listc))