import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import spacy
from collections import defaultdict


sp = '\n\n'
largge = 'en_core_web_lg'
small_ = 'en_core_web_sm'
path = r"D:\PythonDataScience\tweeter"
os.chdir(path)

def print2(*args):
    for arg in args:
        print(arg, end="\n\n")

data = pd.read_csv('nyt2.csv')

print2(list(data.index), data.head())


nlp = spacy.load(largge)

# # Extract the lemma for each token and join
def clean_text(text):

    doc = nlp(text)

    result = " ".join([token.lemma_ for token in doc if (not token.is_punct) and 
            (not token.is_stop) and ('-PRON-' not in token.lemma_ )])

    return result


def text_similarity(text1, text2):

    mydict = defaultdict(list)

    n = 0

    for ind, ele in enumerate(text1):
        
        ele_token = clean_text(ele)

        ele_token2 = nlp(ele_token)
        print("->", n, end= " ", sep= " ")
        for text in text2:
            text_token = clean_text(text)
            text_token2 = nlp(text_token) 
            mydict[ind].append(ele_token2.similarity(text_token2))       
            print(".", end=" ")

            n += 1
    print("Done!", end="\n\n")

    df  = pd.DataFrame(mydict)

    return df


df33 = text_similarity(data.loc[:5, "News_content"], data.loc[:, "News_content"])
all_data = pd.DataFrame()

df33.columns = [f'type{i}' for i in range(len(df33.columns))]
print2(df33)
for lee in df33.columns:
    df33.sort_values(by = lee, inplace=True, ascending=False)
    first5 = df33[lee].head()
    first5.columns = ["similarity"]
    print2(first5)
    all_data = pd.concat([all_data, first5], axis=0, sort=True)

all_data.columns = ["similarity"]
all_data.sort_values(by=["similarity"], ascending=False, inplace=True)
ggg = all_data.loc[~all_data.index.isin([0,1,2,3,4,5])].reset_index()
# print2(ggg[~ggg.index.isin([range(5)])])
print2(ggg.drop_duplicates(subset=["index"]), ggg)


# print2(text_similarity(data.loc[:5, "News_content"], data.loc[:9, "News_content"]))
