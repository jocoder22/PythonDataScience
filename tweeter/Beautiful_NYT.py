#!/usr/bin/env python
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import re
import time
time.sleep(5)


sp = '\n\n'
path = "C:\\Users\\Jose\\Desktop\\PythonDataScience\\tweeter\\"
os.chdir(path)

url2 = 'https://www.nytimes.com/'
url = 'https://www.nytimes.com/2019/02/24/world/europe/pope-vatican-sexual-abuse.html?action=click&module=Top%20Stories&pgtype=Homepage'
res = requests.get(url)
html_doc = res.text

# create a BeautifulSoup object
soup = BeautifulSoup(html_doc, features="lxml")
# pretty_soup = soup.prettify()
# print(soup.prettify())

# Get the title of the NYT webpage
nyt_title = soup.title

# Get nyt's text: nyt_text
nyt_text = soup.get_text()

# Print the title of nyt's webpage to the shell
# print(nyt_title, nyt_text, sep=sp)

# Find all 'a' tags (which define hyperlinks): a_tags
a_tags = soup.find_all('a')

# for link in a_tags:
    # print(link.get('href'))

Nyt_text = []

# # Print the URLs to the shell
# for link in a_tags:
#     hmm = 'https://www.nytimes.com'
#     # getpage = requests.get(link.get('href')).text
#     pagelink = link.get('href')

#     if re.match(r'https://www.nytimes.com', pagelink):
#         getpage = requests.get(pagelink).text
#         soup2 = BeautifulSoup(getpage, features="lxml")

#         for texts in soup2.findAll('p'):
#             Nyt_text.append(texts.text)
#         time.sleep(2)
    # elif len(pagelink)  > 25:
    #     newpagelink = hmm + pagelink
    #     getpage = requests.get(newpagelink).text
    #     soup2 = BeautifulSoup(getpage, features="lxml")
 
    #     for texts in soup2.findAll('p'):
    #         Nyt_text.append(texts.text)
    #         time.sleep(2)
# headline = soup.findAll("span")
# # headline = soup.find("span").contents

headline2 = soup.find("span", {'class': ['css-rs6kf8']})
hd = headline2.text

# # for idx, texts in enumerate(headline):
# #     print(idx, texts.text, texts.attrs)
# # # print(headline)

# # hd = headline[7].text

Nyt_text = [hd]
for texts in soup.findAll('p'):
    Nyt_text.append(texts.text)

data = pd.DataFrame(Nyt_text, columns=['News_content'])
print(data.head())
data.to_csv('nyt2.csv', index=False)