from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC 
from bs4 import BeautifulSoup
import requests
import itertools

session = requests.Session()
url = "http://pycoders.com/archive/"
page = session.get(url).text
bsObj = BeautifulSoup(page)

bsdiv = bsObj.find("div", {"class": "mb-3"})

link = dict()
url2 = "https://pycoders.com"
for a in bsdiv.findAll("a"):
    link[a.contents[0]] = url2 + a.attrs["href"]

dict(itertools.islice(link.items(), 5))


# using selenium WebDriverWait
path = "chromedriver.exe"
driver = webdriver.Chrome(executable_path=path)
driver.get(url)

wait = WebDriverWait(driver, 20, 1)
wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "display_archive")))

bsObj = BeautifulSoup(driver.page_source)
bsdiv2 = bsObj.find("div", {"class": "mb-3"})

link2 = dict()
for a in bsdiv2.findAll("a"):
    link2[a.contents[0]] = url2 + a.attrs["href"]

dict(itertools.islice(link2.items(), 4))