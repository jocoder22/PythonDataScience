import os
import requests
from scrapy import Selector

sp = '\n\n'
url = 'https://www.cdc.gov/nchs/tutorials/NHANES/index_continuous.htm'
res = requests.get(url)
html = res.text 

sel = Selector(text=html)
sll = sel.xpath('//p')[2]  # extract the 3rd element of the selectorList
slla = sel.xpath('//p').extract()
sllf = sel.xpath('//p').extract_first()
print(sll, slla, sllf, sep=sp)