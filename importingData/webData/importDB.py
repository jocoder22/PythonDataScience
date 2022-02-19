#!/usr/bin/env python

import os
from urllib.request import urlretrieve
from sqlalchemy import create_engine, MetaData

path = 'E:\PythonDataScience\importingData\webData'

os.chdir(path)


url = 'http://swcarpentry.github.io/sql-novice-survey/files/survey.db'

urlretrieve(url, 'survey.db')


engine = create_engine('sqlite:///survey.db')
connection = engine.connect()

metadata = MetaData()


for t in metadata.sorted_tables:
    print(t.name)