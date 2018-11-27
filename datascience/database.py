import numpy as np
import pandas as pd
import sqlite3


droptable = "DROP TABLE IF EXISTS temp_data;"
createtable = "CREATE TABLE WeatherData \
                (Date INTEGER, City VARCHAR(40), Season VARCHAR(40) \
                Temperature REAL, Grade INTEGER);"
connection = sqlite3.connect("Test01.db")
connection.execute(droptable)
connection.execute(createtable)
connection.commit()


# get the data
mydata = [(20181109, "New York", "Winter", 23.9, 2),
          (20181112, "Brookly", "Winter", 25.0, 2),
          (20181114, "Boston",  "Winter", 33.9, 3),
          (20181116, "Queens", "Winter", 15.9, 1),
          (20181114, "Newark",  "Winter", 13.9, 1),
          (20181116, "Hicksvill", "Winter", 10.9, 1)]
insert_query = "INSERT INTO WeatherData VALUES(?, ?, ?, ?)"
connection.executemany(insert_query, mydata)
connection.commit()
