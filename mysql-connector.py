import mysql.connector

conn = mysql.connector.connect(
            host="localhost",
            user="josh",
            passwd="pass23")

cur = conn.cursor()
cur.execute("CREATE DATABASE mydbase44")
cur.execute("use mydbase44")




