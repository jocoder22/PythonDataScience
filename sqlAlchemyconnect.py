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



data = pd.read_csv('PopPyramids.csv')
data.head()

# change column name to lower case and replace space with underscore
data.columns = pd.Index(pd.Series(data.columns).map(lambda x: x.lower().replace(" ","_")))
data.index.rename([n.lower() for n in data.index.names], inplace=True)
data.head()


make_table = """CREATE TABLE 'populations'(
                'region' varchar(20),
                'year' int(4),
                'both_sexes_population' bigint(20),
                'male_population' bigint(20),
                'female_population' bigint(20),
                'percent_both_sexes' double,
                'percent_male' double,
                'percent_female' double,
                'sex_ratio' double,
                'age' char(5) NOT NULL,
                'country' char(28) NOT NULL,
                PRIMARY KEY ('country', 'year', 'age')
                );"""

conn.execute(make_table)
data.to_sql('population',con=conn, if_exists='append')




conn.close()