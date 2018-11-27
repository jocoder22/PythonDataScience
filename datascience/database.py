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
