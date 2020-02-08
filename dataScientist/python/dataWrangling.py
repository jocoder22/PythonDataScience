import numpy as np
import pandas as pd

def print2(*args):
    for arg in args:
        print(arg, end="\n\n")
        
def countInline(dataframe, c1, c3, lookuplist):
    from collections import defaultdict
    mydict = defaultdict(int)
    
    for i in range(dataframe.shape[0]):
        for answer in lookuplist:
            if dataframe[c1][i].split("; "):
                mydict[answer] += int(dataframe[c2][i])
    
    mydata = pd.DataFrame(pd.Series(mydict)).reset_index()
    mydata.columns = [c1, c2]
    mydata.sort_values(c2, ascending=False, inplace=True)
    return mydata
    

path = r"C:\Users\okigb\Downloads\survey-results-public.csv"
path2 = r"C:\Users\okigb\Downloads\survey-results-schema.csv"

df = pd.read_csv(path)

df2 = df.CousinEducation.value_counts().reset_index()
df2.rename(columns={"index":"Method", "CousinEducation":"Count"}, inplace=True)

print2(df.CousinEducation, df.columns, df.shape, df2.head())

schema = pd.read_csv(path2)
print2(schema, list(schema[schema.Column == "CousinEducation"]["Question"]))

answerlist = []
for i in range(df2.shape[0]):
    for k in  df2['Method'][i].split("; "):
        if k not in answerlist:
            answerlist.append(k)
            
print2(answerlist)

