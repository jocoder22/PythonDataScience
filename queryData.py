import pymysql
from sqlalchemy import create_engine
from pandas import pd
from pandas import DataFrame


def connector_string(user, passwd, host, dbase):
    return "mysql+pymysql://" + user + ";" + passwd + "@" + host + "/" + dbase


conn = create_engine(connector_string('root',
                     pssd, 'localhost', 'pop')).connect()


# query the database
pop1_ = pd.read_sql('SELECT * FROM pop  WHERE country ="United States" AND year=2018;',
                    con=conn,
                    index_col=["country", "year", "age"])

            