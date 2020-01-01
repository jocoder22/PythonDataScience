import requests
from bs4 import BeautifulSoup as Bs

def print2(*args):
    for arg in args:
        print(arg, end='\n\n')
        
              
sp = {"sep":"\n\n", "end":"\n\n"}  


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
