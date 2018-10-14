# import pymysql
# # import MySQLdb
# # pip install mysqlclient
# pymysql.install_as_MySQLdb()
# import MySQLdb

# db = MySQLdb.connect(host="localhost",  # your host 
#                      user="root",       # username
#                      passwd="root",     # password
#                      db="pythonspot") 

# db = MySQLdb.connect(user="my-username", passwd="my-password", 
#                      host="localhost", db="my-databasename")
# cursor = db.cursor()

import pymysql as sql
import mysql.connector

conn = sql.connect(host='localhost',
                   user='root',
                   password='pass23',
                   db='mysql')
cur = conn.cursor()

cur.execute('show databases;')  # 8
cur.execute('use world;')  # 0
cur.execute('show tables;')  # 3


# Close the connection
cur.close()
conn.close()

conn = sql.connect(host='localhost',
                   user='josh',
                   password='kelly45',
                   db='house')

conn.close()




# Establish connection
conn = sql.connect(host='localhost',
                   user='root',
                   password='pass23',
                   db='poppyramids')

cur = conn.cursor()

make_table = """CREATE TABLE populations(
                    region varchar(20),
                    year int(4),
                    both_sexes_population bigint(20),
                    male_population bigint(20),
                    female_population bigint(20),
                    percent_both_sexes double,
                    percent_male double,
                    percent_female double,
                    sex_ratio double,
                    age char(5) NOT NULL,
                    country char(28) NOT NULL,
                    PRIMARY KEY ('country', 'year', 'age')
                    );"""

cur.execute(make_table)


conn = mysql.connector.connect(host='localhost',
                   user='root',
                   password='pass23')


mydb = mysql.connector.connect(
  host="localhost",
  user="josh",
  password="kelly45")