import os
import pandas as pd 
from pandas import DataFrame
from sqlalchemy import create_engine

os.chdir("C:/Users/Jose/Documents/PythonDataScience1/Code/Code/Section 1")

def sql_str_generator(user, pword, host, dbase):
    """
    Return string use for connection
    """
    return "mysql+pymysql://" + user + ":" + pword + "@" + host + "/" + dbase


connstr = sql_str_generator("root","pass23","localhost","world")
conn = create_engine(connstr).connect()
conn.close()


data = pd.read_csv('PopPyramids.csv')
data.head()
