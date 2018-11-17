from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC 
from bs4 import BeautifulSoup
import requests

session = requests.Session()
url = "http://pycoders.com/achive/"
page = session.get(url).text
bsObj = BeautifulSoup(page)

link1 = dict()
for a in bsObj.findAll("a"):
    link1[a.contents[0]] = a.attrs["href"]

link1