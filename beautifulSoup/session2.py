import requests
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
