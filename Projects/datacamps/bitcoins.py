#!/usr/bin/env python
import os
import json
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
from requests import Request, Session
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# plt.style.use('ggplot')
from collections import defaultdict

path = 'C:\\Users\\okigboo\\Desktop\\PythonDataScience\\Projects\\datacamps\\'
os.chdir(path)



url2 = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'


ky = "e8c48d14-p034-8a2a-4c39-870c-8cddf5db70b3"
parameters = {
    'start': '1',
    'limit': '5000',
    'convert': 'USD',
}
headers = {
    'Accepts': 'application/json',
    'X-CMC_PRO_API_KEY': ky,
}

session = Session()
session.headers.update(headers)

try:
    response = session.get(url2, params=parameters)
    data2 = json.loads(response.text)
    print(data2[:2])
except (ConnectionError, Timeout, TooManyRedirects) as e:
    print(e)



data = data2['data']
# mylist = list()
mydict = defaultdict(list)
for i, item in enumerate(data):
    for k, v in item.items():
        if k == 'quote':
            mydict['price'].append(data[i][k]['USD']['price'])
            mydict['volume_24h'].append(data[i][k]['USD']['volume_24h'])
            mydict['percent_change_1h'].append(data[i][k]['USD']['percent_change_1h'])
            mydict['percent_change_24h'].append(data[i][k]['USD']['percent_change_24h'])
            mydict['percent_change_7d'].append(data[i][k]['USD']['percent_change_7d'])
            mydict['market_cap'].append(data[i][k]['USD']['market_cap'])
        else:
            mydict[k].append(v)


data22 = pd.DataFrame(mydict)
data22.columns
data22.head()

data22.to_csv('crypto.csv', index=False)
