
import pymysql
import MySQLdb
# pip install mysqlclient
pymysql.install_as_MySQLdb()


db = MySQLdb.connect(host="localhost",  # your host 
                     user="root",       # username
                     passwd="root",     # password
                     db="pythonspot") 


