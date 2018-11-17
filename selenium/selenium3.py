from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC 
from bs4 import BeautifulSoup
import requests

session = requests.Session()
url = "http://pycoders.com/archive/"
page = session.get(url).text
bsObj = BeautifulSoup(page)

link1 = dict()
for a in bsObj.findAll("a"):
    link1[a.contents[0]] = a.attrs["href"]

link1

path = "chromedriver.exe"
driver = webdriver.Chrome(executable_path=path)
driver.get(url)

wait = WebDriverWait(driver, 60, 1)
wait.until(EC.presence_of_all_elements_located((By.CLASS_NAME, "display_archive")))

bsObj = BeautifulSoup(driver.page_source)

link2 = dict()
url2 = "https://pycoders.com"
for a in bsObj.findAll("a"):
    link2[a.contents[0]] = url2 + a.attrs["href"]

link2

import itertools
dict(itertools.islice(link2.items(), 2))