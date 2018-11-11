import requests
import pandas as pd 
from datetime import datetime
from time import sleep
from pandas import DataFrame
from bs4 import BeautifulSoup as Bs


# Begin a session
session = requests.Session()
# to see more https://www.whatismybrowser.com/

url = "https://en.wikipedia.org/wiki/List_of_Nobel_laureates"

page = session.get(url).text


# make a BeautifulSoup object
nobelist = Bs(page)
nobeltable = nobelist.find("table", {"class":['wikitable', 'sortable']})

tabledata = nobeltable.findAll("td")

for data in tabledata:
    print(data.a)

# Add the information inside a dictionary
links = dict()

for alinks in tabledata:
    if alinks.a != None:
        links[alinks.a.contents[0]] = alinks.a.attrs["href"]
links


# Create baseurl
baseurl = "https://en.wikipedia.org"
baseurl + links['William Nordhaus']  # 'https://en.wikipedia.org/wiki/William_Nordhaus'


winnerpage = session.get(baseurl + links['William Nordhaus']).text
winnerObj = Bs(winnerpage)
winnerObj.find("table", {"class": ["infobox", "biography", "vcard"]})
winnerObj.find("span", {"class": "bday"})


datetime.strptime(winnerObj.find("span", {"class": "bday"}).contents[0], "%Y-%m-%d")
# datetime.datetime(1941, 5, 31, 0, 0)