from selenium import webdriver
from time import sleep

path = "C:/webdrivers/chromedriver.exe"
driver = webdriver.Chrome(executable_path=path)
sleep(12)
driver.get("http://pycoders.com/archive/")
sleep(12)
driver.close()