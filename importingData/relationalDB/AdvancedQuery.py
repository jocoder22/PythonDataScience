#!/usr/bin/env python
# Import packages
import os
from sqlalchemy import create_engine, MetaData
import pandas as pd
import matplotlib.pyplot as plt


path = 'E:\DatacampDataScientistPython'
os.chdir(path)

# Create engine: engine
engine = create_engine('sqlite:///Chinook.sqlite')

metadata = MetaData()


for t in metadata.sorted_tables:
    print(t.name)





# Open engine in context manager
# Perform query and save results to DataFrame: df
with engine.connect() as con:
    rs = con.execute("SELECT Title, Name FROM Album INNER JOIN Artist WHERE Album.ArtistID = Artist.ArtistID")
    df = pd.DataFrame(rs.fetchall())
    df.columns = rs.keys()

# Print head of DataFrame df
print(df.head())


# # Execute query and store records in DataFrame: df
df = pd.read_sql_query("""SELECT *
                        FROM PlaylistTrack 
                        INNER JOIN Track 
                        on PlaylistTrack.TrackId = Track.TrackId
                        WHERE Milliseconds < 250000
                    """, engine)

# Print head of DataFrame
print(df.head())


