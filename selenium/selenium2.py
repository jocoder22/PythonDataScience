from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from time import sleep

driver = webdriver.Chrome()
driver.get("https://google.com")

sleep(5)
txtsearch = webdriver.find_element_by_id("1st-ib")
btnsearch = webdriver.find_element_by_name("btn")

txtsearch.send_keys("selenium")
sleep(5)
txtsearch.send_keys(Keys.ECAPE)
btnsearch.click()
sleep(5)