#!/usr/bin/env python

import os
from urllib.request import urlretrieve

os.chdir('c:/Users/Jose/Desktop/PythonDataScience/importingData/webData/')

url = 'http://swcarpentry.github.io/sql-novice-survey/files/survey.db'

urlretrieve(url, 'survey.db')