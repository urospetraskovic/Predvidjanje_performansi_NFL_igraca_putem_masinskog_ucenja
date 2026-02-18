from selenium import webdriver
from selenium.common.exceptions import TimeoutException
import time

url = "https://www.pro-football-reference.com/players/A/AlleJo02.htm"

print("Testing Selenium with Chrome...")
try:
    options = webdriver.ChromeOptions()
    options.add_argument('--start-maximized')
    
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    time.sleep(3)
    print(f"Page loaded successfully")
    print(f"Page title: {driver.title}")
    driver.quit()
except Exception as e:
    print(f"Error: {e}")
