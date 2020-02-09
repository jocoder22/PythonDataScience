import numpy as np
import pandas as pd

def print2(*args):
    for arg in args:
        print(arg, end="\n\n")

        
def countInline(dataframe, c1):
    from collections import defaultdict
    
    mydict = defaultdict(int)
    satisfy = defaultdict(int)
    
    factors = ["CareerSatisfaction", "JobSatisfaction","StackOverflowSatisfaction"]
    factor2 = ["CareerSatisfaction", "JobSatisfaction","StackOverflowSatisfaction", "CousinEducation"]

    
    dss = dataframe.copy()
    dss.dropna(subset=factor2, inplace=True)
    
    for i in dss.index:
        for meth in dss[c1][i].split("; "):
            satisfy[meth] += dss.loc[i, factors].sum() / 3
            mydict[meth] += 1
            
    mydata = pd.DataFrame(pd.Series(mydict)).reset_index()
    mydata2 = pd.DataFrame(pd.Series(satisfy)).reset_index()
    
    mydata.columns =  [c1, "Count"]
    mydata2.columns = [c1, "Satisfaction"] 
    
    allmydata = pd.merge(mydata, mydata2)
    # allmydata["Percentage"] = allmydata['Count']/np.sum(allmydata.Count) 
    # allmydata["Satisfaction"] = allmydata['Satisfaction']/np.sum(allmydata.Satisfaction) * 100
    allmydata["Satisfaction"] = allmydata['Satisfaction']/allmydata.Count 
    
    allmydata.sort_values("Satisfaction", ascending=False, inplace=True)
  
    return allmydata


def salaryFactor(dataframe, c1, c2):
    from collections import defaultdict
    
    mydict = defaultdict(int)
    pay = defaultdict(int)
   
    dss = dataframe.copy()
    dss.dropna(subset=[c1, c2], inplace=True)
    
    for i in dss.index:
        for meth in dss[c1][i].split("; "):
            mydict[meth] += 1
            pay[meth] += dss.loc[i, c2]
            
    mydata = pd.DataFrame(pd.Series(mydict)).reset_index()
    mydata4 = pd.DataFrame(pd.Series(pay)).reset_index()
    
    mydata.columns = [c1, "Count"]
    mydata4.columns = [c1, c2]
    
    allmydata = pd.merge(mydata4, mydata)
    allmydata['Ave_salary'] = allmydata[c2] / allmydata["Count"]
  
    allmydata.sort_values("Ave_salary", ascending=False, inplace=True)
  
    return allmydata
    
    

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

data4 = countInline(df, "CousinEducation")


