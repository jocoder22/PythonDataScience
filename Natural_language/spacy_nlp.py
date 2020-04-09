#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from spacy import displacy
from pathlib import Path

# pip install pyICU


sp = '\n\n'
largge = 'en_core_web_lg'
small_ = 'en_core_web_sm'
# path = "C:\\Users\\Jose\\Desktop\\PythonDataScience\\tweeter\\"
# path = 'C:\\Users\\okigboo\\Desktop\\PythonDataScience\\tweeter\\'
path = r"D:\PythonDataScience\tweeter"
os.chdir(path)

data = pd.read_csv('nyt2.csv')

text_clean = []

mytext = str()

for text in data['News_content']:
    mytext += text + " "
    
# Instantiate the English model: nlp
nlp = spacy.load(largge)


for text in data['News_content']:
    spacyd = nlp(text)
#     displacy.serve(spacyd, style='ent')
#     see localhost:xxxx
    for ent in spacyd.ents:
        print(text, sep=sp)
        print(ent.label_, ent.text)
        print(" ", sep=sp)
 


# pp = 'C:\\Users\\Jose\\Desktop\\PythonDataScience\\Natural_language\\'
pp = r"D:\PythonDataScience\Natural_language"

nlp = spacy.load(largge)
sentences = ["This is an example.", "This is another one."]
for sent in sentences:
    doc = nlp(sent)
    svg = displacy.render(doc, style='dep')
    file_name = '-'.join([w.text for w in doc if not w.is_punct]) + '.svg'
    output_path = Path(pp + file_name)
    output_path.open('w', encoding='utf-8').write(svg)



text = """This is new worthy, although Google bought the compnay from behind. The company made a late push into  software and hardware. Samsung, Hawei and Apple, available on iPhones, and Amazon’s Alexa software, are in Washington DC to lobby for the lease which runs on its Echo and Dot devices, have clear leads  and superiority in consumer adoption."""

nlp = spacy.load(largge)
doc = nlp(text)
displacy.serve(doc, style='ent')
