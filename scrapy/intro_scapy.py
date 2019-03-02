import os
import requests
from scrapy import Selector

def number_of_element(Xpath, inputtext):
    sp = '\n\n'
    sel = Selector(text=inputtext)
    print(f"Number of selected element(s): {len(sel.xpath(Xpath))} elements", end=sp)


def preview_result(Xpath, inputtext):
    sp = '\n\n'
    sel = Selector(text=inputtext)
    result = sel.xpath(Xpath).extract()
    n = len(result)
    for idx, element in enumerate(result[:min(4,n)], start=1):
        print(f"Element {idx}: {element}", end=sp)


sp = '\n\n'
url = 'https://www.cdc.gov/nchs/tutorials/NHANES/index_continuous.htm'
res = requests.get(url)
html = res.text 

xpath = '//p'
sel = Selector(text=html)
sll = sel.xpath('//p')[2].extract()  # extract the 3rd element (here paragrph) of the selectorList
sll_ = sel.xpath('//p')  # without extract(), the selectorList give a 36 line preview of the paragraph
slla = sel.xpath('//p').extract()
sllf = sel.xpath('//p').extract_first()



# print(sll, slla, sllf, sep=sp)

print(number_of_element(xpath, html), preview_result(xpath, html), sep=sp)