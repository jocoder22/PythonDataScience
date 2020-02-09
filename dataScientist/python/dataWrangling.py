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

