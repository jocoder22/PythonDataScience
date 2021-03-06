import numpy as np
import pandas as pd

def print2(*args):
    for arg in args:
        print(arg, end="\n\n")
        
path = r"C:\Users\okigb\Downloads\survey-results-public.csv"
path2 = r"C:\Users\okigb\Downloads\survey-results-schema.csv"

df = pd.read_csv(path)
schema = pd.read_csv(path2)
print2(df.head(), schema.head())


question = list(schema[schema["Column"] == "JobSatisfaction"]["Question"])[0]
prop = sum(df.JobSatisfaction.isnull()) / df.shape[0]
prop2 = df.JobSatisfaction.isnull().mean()
prop3 = (df.JobSatisfaction.isnull() == False).mean()
print2(question, prop, prop2, prop3)

ej_sum = df.groupby("EmploymentStatus")["JobSatisfaction"].sum().reset_index()
ej_count = df.groupby("EmploymentStatus")["JobSatisfaction"].count().reset_index()
ej_count.columns = ["EmploymentStatus", "Count"]
ejdata = pd.merge(ej_sum, ej_count)
ejdata["percent"] = ejdata.JobSatisfaction / ejdata.Count
ejdata.sort_values("percent", ascending=False, inplace=True)




dfff = df.groupby("CompanySize")["JobSatisfaction"].sum().reset_index()
dff2 = df.groupby("CompanySize")["JobSatisfaction"].count().reset_index()
dff2.columns = ["CompanySize", "Count"]
alldata = pd.merge(dff2,dfff)
alldata["percent"] = alldata.JobSatisfaction / alldata.Count
alldata.sort_values("percent", ascending=False, inplace=True)
print2(alldata, ejdata)


df2 = df.groupby("CompanySize").mean()["JobSatisfaction"].sort_values(ascending=False)
ej2 = df.groupby("EmploymentStatus").mean()["JobSatisfaction"].sort_values(ascending=False)
homeRemote = df.groupby("HomeRemote").mean()["JobSatisfaction"].sort_values(ascending=False)
progHobby = df.groupby("ProgramHobby").mean()["JobSatisfaction"].sort_values(ascending=False)
formalEdu = df.groupby("FormalEducation").mean()["JobSatisfaction"].sort_values(ascending=False)

print2(df2, ej2, homeRemote, progHobby, formalEdu)

print(df.notnull().all().sum(), df.shape[1], sep="\n\n")
# print(df.notnull().all().values)

lls = []
for i,v  in enumerate(df.notnull().all().values):
    if v == True:
        lls.append(df.notnull().all().index[i])

print(lls)