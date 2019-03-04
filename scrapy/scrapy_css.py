#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scrapy
from scrapy import Selector
from scrapy.crawler import CrawlerProcess
# plt.style.use('ggplot')


url = 'https://en.wikipedia.org/wiki/Main_Page'

class RResponse_css(scrapy.Spider):
    name = 'rresponse_css'
    start_urls = [url]


process = CrawlerProcess()
process.crawl(RResponse_css)
process.start()

print(response.url)

