import requests
from bs4 import BeautifulSoup as Bs
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
nobelist.find_all("a", {"class" : "internal"})[0].attrs["href"]
nobelist.find_all("a", {"class" : "internal"})[1].attrs["href"]
nobelist.find_all("a", {"class" : "internal"})[1].attrs["title"]

nobelist.find("h1")
nobelist.find("h1").contents



