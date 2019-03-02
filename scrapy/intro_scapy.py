import os
import requests
from scrapy import Selector

sp = '\n\n'
url = 'https://www.cdc.gov/nchs/tutorials/NHANES/index_continuous.htm'
res = requests.get(url)
html = res.text 

sel = Selector(text=html)
sll = sel.xpath('//p')[2]
print(sll)