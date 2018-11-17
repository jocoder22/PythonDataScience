from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from time import sleep
from PIL import Image

driver = webdriver.Chrome()
driver.get("https://google.com")

sleep(5)
txtsearch = driver.find_element_by_id("q")
btnsearch = driver.find_element_by_name("btnK")

txtsearch.send_keys("selenium")
sleep(3)
txtsearch.send_keys(Keys.ESCAPE)
btnsearch.click()
sleep(2)

lst_aRes = driver.find_element_by_xpath("//div[@id='extrares']//a")

num_common = len(lst_aRes)

term = set()

for i in range(num_common):
    term.add(lst_aRes[i].text)
    lst_aRes[i].click()
    sleep(4)
    lst_aChildRes = driver.find_element_by_xpath("//div[@id='extrares']//a")

    for a in lst_aChildRes:
        term.add(a.text)
    driver.back()

    lst_aRes = driver.find_element_by_xpath("//div[@id='extrares']//a")
    sleep(2)
        

links = list()
for a in lst_aRes:
    links.append(a.get_attribute("href"))

links

term

driver.get_screenshot_as_file("screenshot.png")
Image.open("screenshot.png")
driver.close()