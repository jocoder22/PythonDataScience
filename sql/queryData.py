import pymysql
from sqlalchemy import create_engine
from pandas import pd
from pandas import DataFrame


def connector_string(user, passwd, host, dbase):
    return "mysql+pymysql://" + user + ";" + passwd + "@" + host + "/" + dbase


conn = create_engine(connector_string('root',
                     pssd, 'localhost', 'pop')).connect()


# query the database
pop1_ = pd.read_sql('''SELECT * FROM pop  WHERE country="United States" AND
                    year=2018;''',
                    con=conn,
                    index_col=["country", "year", "age"])

pop1_


# Add other queries
pd.read_sql('SELECT * FROM pop;', con=conn,
            index_col=['country', 'year', 'age'])

pd.read_sql('''SELECT country, year, both_sexes_population
            FROM pop
            WHERE age = "Total" AND (year=2017 or year=2018);''',
            con=conn, index_col=["country", "year"])

pd.read_sql('SELECT country FROM pop;', con=conn)

pd.read_sql('SELECT DISTINCT country FROM pop;', con=conn)


conn.close()
