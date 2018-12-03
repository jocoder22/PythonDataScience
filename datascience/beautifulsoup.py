from urllib.request import urlopen
from bs4 import BeautifulSoup

url = 'https://en.wikipedia.org/wiki/William_Shakespeare'
response = urlopen(url)
bsObject = BeautifulSoup(response, 'html.parser')
bsObject.title

section = bsObject.find_all(id='catlinks')[0]
for catlink in section.find_all("a")[1:]:
    print(catlink.get("title"), "->", catlink.get("href"))
