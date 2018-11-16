from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from time import sleep

driver = webdriver.Chrome()
driver.get("https://google.com")

sleep(5)
txtsearch = webdriver.find_element_by_id("q")
btnsearch = webdriver.find_element_by_name("btnk")

txtsearch.send_keys("selenium")
sleep(3)
txtsearch.send_keys(Keys.ESCAPE)
btnsearch.click()
sleep(2)