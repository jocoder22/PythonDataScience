from urllib.request import urlopen
from bs4 import BeautifulSoup

url = 'https://en.wikipedia.org/wiki/William_Shakespeare'
response = urlopen(url)
bsObject = BeautifulSoup(response, 'html.parser')
bsObject.title

section = bsObject.find_all(id='catlinks')[0]
for catlink in section.find_all("a")[1:]:
    print(catlink.get("title"), "->", catlink.get("href"))


section2 = bsObject.find_all(id='mw-normal-catlinks')[0]
for catlink in section2.find_all("a")[1:]:
    print(catlink.get("title"), "->", catlink.get("href")) 


 bs2 = section = soup.find_all("div" ,{"class" : "refbegin"})[0]
 
