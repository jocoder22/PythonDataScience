import requests
from bs4 import BeautifulSoup as Bs

def print2(*args):
    for arg in args:
        print(arg, end='\n\n')
        
        
sp = {"sep":"\n\n", "end":"\n\n"}  


session = requests.Session()
# to see more https://www.whatismybrowser.com/

url = "https://en.wikipedia.org/wiki/List_of_Nobel_laureates"

header = {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
          "Accept-Language": "en-US,en;q=0.5",
          "Connection": "keep-alive",
          "Referrer": "https://www.goggle.com/",
          "User-Agent": "Mozilla/5.0 (X11; Ubuntu; Linux i686, rv:54.8) Gecko/20180101 Firefox/54.0"}

# page = session.get(url, headers=header).text
page = session.get(url).text
print(page)

nobelist = Bs(page)
nobelist.find("a")
nobelist.find("div")

# specific links
nobelist.find_all("a", {"class" : "internal"})
"""[<a class="internal" href="/wiki/File:Nobel_Prize.png" title="Enlarge"></a>,
    <a class="internal" href="/wiki/File:Nobel_Prize_winners_2012.jpg" title="Enlarge"></a>]
"""
nobelist.find_all("a", {"class" : "internal"})[0].attrs["href"]
nobelist.find_all("a", {"class" : "internal"})[1].attrs["href"]
nobelist.find_all("a", {"class" : "internal"})[1].attrs["title"]

nobelist.find("h1")
# [<h1 class="firstHeading" id="firstHeading" lang="en">List of Nobel laureates</h1>]
nobelist.find("h1").contents  # ['List of Nobel laureates']


# iterate over the search result
alist = nobelist.findAll("a", {"class" : "internal"})
type(alist)  # <class 'bs4.element.ResultSet'>

for i in range(len(alist)):
    alist[i].attrs["href"]


nobelist.table
nobelist.table.attrs["class"]
nobelist.findAll("table", {"class":['wikitable', 'sortable']})


children = nobelist.table.children  # <list_iterator object at 0x048306B0>

tbody = nobelist.table.tbody # the table body without the header
count = 1
for child in tbody:  # iterate throught the rows of the table
    print("\nNode: Number %s \n-----" % count)
    print("Name: %s \n-----" % child.name)
    print(child)
    count += 1

count = 1
for child in nobelist.table.findAll("tr"):  # iterate throught the rows of the table
    print("\nNode: Number %s \n-----" % count)
    print("Name: %s \n-----" % child.name)
    print(child)
    for i in range(4): print()
    count += 1


nobelist.table.tr
nobelist.table.findAll("td")
nobelist.table.parent

