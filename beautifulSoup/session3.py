import requests
import pandas as pd 
from datetime import datetime
from time import sleep
from pandas import DataFrame
from bs4 import BeautifulSoup as Bs
import matplotlib.pyplot as plt


# Begin a session
session = requests.Session()
# to see more https://www.whatismybrowser.com/

url = "https://en.wikipedia.org/wiki/List_of_Nobel_laureates"

page = session.get(url).text


# make a BeautifulSoup object
nobelist = Bs(page)
# nobeltable = nobelist.find("table", {"class":['wikitable', 'sortable']})

# tabledata = nobeltable.findAll("td")

# for data in tabledata:
#     print(data.a)


nobeltable2 = nobelist.find("table", {"class":['wikitable', 'sortable']})

tabledata = nobeltable2.findAll("span", {"class": "fn"})
tabledata

for data in tabledata:
    print(data.a)


link2 = dict()
for item in nobeltable2.findAll("td"):
        if item.a != None and item.a.attrs["href"][0:6] == "/wiki/":
                link2[item.a.contents[0]] = item.a.attrs["href"]

len(link2)


# Add the information inside a dictionary
links = dict()

for alinks in tabledata:
    if alinks.a != None:
        links[alinks.a.contents[0]] = alinks.a.attrs["href"]

len(links)

print(links)


# Create baseurl
baseurl = "https://en.wikipedia.org"
baseurl + links['William Nordhaus']  # 'https://en.wikipedia.org/wiki/William_Nordhaus'


# scrap linked webpage
winnerpage = session.get(baseurl + links['William Nordhaus']).text
winnerObj = Bs(winnerpage)
winnerObj.find("table", {"class": ["infobox", "biography", "vcard"]})
winnerObj.find("span", {"class": "bday"})

# Extract the needed info: birthday
datetime.strptime(winnerObj.find("span", {"class": "bday"}).contents[0], "%Y-%m-%d")
# datetime.datetime(1941, 5, 31, 0, 0)

datadict = dict()
for name, link in links.items():
    sleep(2) # wait for ten seconds between pages
    print("Fetching: " + name)
    winnerpg = session.get(baseurl + links[name]).text
    winnerObj = Bs(winnerpg)
    winnerbdate = winnerObj.find("span", {"class": "bday"})
    if winnerbdate != None:
        try:
            bthday = datetime.strptime(winnerbdate.contents[0], "%Y-%m-%d")
            datadict[name] = {"Year": bthday.year,
                              "Month": bthday.month,
                              "Day": bthday.day}
        except ValueError:
            pass

print(datadict)
NobelData = DataFrame(datadict).T
print(NobelData)

# plot the dataframe
NobelData.Month.value_counts().sort_index().plot(kind='hist')
plt.show()

NobelData.Month.value_counts().sort_index().plot(kind='bar')
plt.show()

