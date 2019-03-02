import os
import requests
from scrapy import selector

url = 'https://www.cdc.gov/nchs/tutorials/NHANES/index_continuous.htm'
res = requests.get(url)
html = res.text 
