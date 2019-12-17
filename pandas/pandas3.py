#!/usr/bin/env python
import numpy as np
import pandas as pd
from datetime import date
from datetime import timedelta
from string import ascii_letters, digits, ascii_lowercase

users = 'Bob John Peter James Anna Mary'.split()
password = [''.join(np.random.choice(list(ascii_letters + digits), 12)) for x in range(len(users))]
listDate = [date.today() - timedelta(days=int(x)) for x in np.random.randint(2, 100, len(users))]
FirstName = ["".join(np.random.choice(list(ascii_lowercase), 6)).capitalize() for x in range(len(users))]
LastName = ["".join(np.random.choice(list(ascii_lowercase), 8)).capitalize() for x in range(len(users))]
print(password, listDate, FirstName, LastName)