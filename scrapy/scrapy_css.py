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

# xpath = '/html/body/span[1]//a'
# Create the CSS Locator string equivalent to the XPath
css_locator = 'html > body > span:nth-of-type(1) a'

# xpath = '//div[@id="uid"]/span//h4'
# Create the CSS Locator string equivalent to the XPath
css_locator = 'div#uid > span h4'
css_locator = 'div.course-block > a'
# Create the CSS Locator to all children of the element whose id is uid
css_locator = "#uid > *"


# Select all hyperlinks of div elements belonging to class "course-block"
course_as = sel.css('div.course-block > a')

# Selecting all href attributes chaining with css
hrefs_from_css = course_as.css('::attr(href)')

# Selecting all href attributes chaining with xpath
hrefs_from_xpath = course_as.xpath('./@href')


# Create an XPath string to the desired text.
xpath = '//p[@id="p3"]/text()'

# Create a CSS Locator string to the desired text.
css_locator = 'p#p3::text'
