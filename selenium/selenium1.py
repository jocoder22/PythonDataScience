from selenium import webdriver
from time import sleep

path = "C:/webdrivers/chromedriver.exe"
driver = webdriver.Chrome(executable_path=path)
sleep(12)
driver.get("http://pycoders.com/archive/")
sleep(12)
print(driver.title)
print(driver.current_url)
driver.close()


path2 = "C:/webdrivers/geckodriver.exe"
driver = webdriver.Firefox(executable_path=path2)
driver.get("http://www.python.org")
assert "Python" in driver.title
elem = driver.find_element_by_name("q")
elem.clear()
elem.send_keys("pycon")
elem.send_keys(Keys.RETURN)
assert "No results found." not in driver.page_source
print(driver.title)
print(driver.current_url)
driver.close()