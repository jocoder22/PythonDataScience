#!/usr/bin/env python

import requests
from bs4 import BeautifulSoup

# Specify the url: url
url2 = "http://www.datacamp.com/teach/documentation"
url = 'https://www.crummy.com/software/BeautifulSoup/'

# Packages the request, send the request and catch the response: r
r = requests.get(url)

# Extract the response: text
html_doc = r.text

# create a BeautifulSoup object
soup = BeautifulSoup(html_doc)

print(soup.prettify())
