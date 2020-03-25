#!/usr/bin/env python

import os
from urllib.request import urlretrieve

path = 'D:\PythonDataScience\importingData\webData'

os.chdir(path)


url = 'http://swcarpentry.github.io/sql-novice-survey/files/survey.db'

urlretrieve(url, 'survey.db')