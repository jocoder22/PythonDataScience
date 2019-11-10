import os
import pandas as pd
import mypackage
from collections import Counter

path = "D:\PythonDataScience\importingData\localData"
sp = {"sep":"\n\n", "end":"\n\n"}

os.chdir(path)
print(os.getcwd(), **sp)

data = pd.read_csv("portfolios.csv", parse_dates = True, index_col= 0)


mypackage.print2(data)


word = 'pope francis end landmark meeting calling battle fight sexual abuse'

wtt = mypackage.TextTokenizer(word)
print(wtt.wordcount, **sp)
print(wtt.greet, **sp)
print(mypackage.tokenize(word), **sp)
path = r'D:\PythonDataScience\MachineLearning\FeatureEngineering'
os.chdir(path)
data = pd.read_csv('textdata.csv', compression='gzip')

# preprocess the text
text_cleanList = []
text_cleanstring = []

for text in data['News_content']:
    text_cleanstring.append(Counter(mypackage.tokenize(text)))
    text_cleanList.extend(mypackage.tokenize(text))
    
result = " ".join(text.strip() for text in data['News_content'])


# form a TextTokenizer Object
wordanalysed = mypackage.TextTokenizer(result)
print(wordanalysed.wordcount, **sp)
wordanalysed.plot_count(n_common=7)


# print(text_cleanstring, sep='\n\n')
# print(text_cleanList, sep='\n\n')


# Sum word_counts using countwordtokens
wordcounts = mypackage.countwordtokens(text_cleanstring)

# Plot wordcounts using plotcount 
mml = mypackage.plotcount(wordcounts)
print(mml, **sp)