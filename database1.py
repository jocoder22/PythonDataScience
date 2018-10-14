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


