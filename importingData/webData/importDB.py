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

# https://assets.datacamp.com/production/repositories/1069/datasets/578834f5908e3b2fa575429a287586d1eaeb2e54/countries2.zip
# https://assets.datacamp.com/production/repositories/1069/datasets/5aba4b2d25e3025de97d9715a022f5c24b74f347/leaders2.zip
# https://assets.datacamp.com/production/repositories/1069/datasets/379b79c12b968edafe24e4bc02fae89d090a9490/diagrams.zip