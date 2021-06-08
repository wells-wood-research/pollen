from selenium import webdriver
from bs4 import BeautifulSoup
from selenium.webdriver.chrome.options import Options
import requests

options = Options()
options.add_experimental_option("excludeSwitches", ["enable-logging"])
driver = webdriver.Chrome(
    executable_path=r"C:/Users/pinto/OneDrive - University of Edinburgh/coding/chromedriver",
    options=options,
)

driver.get(
    "http://www-saps.plantsci.cam.ac.uk/pollen/pollen/pages/Abies%20nordmanniana1.htm"
)
# No list to iterate over here, so used while loop to proccess dataset
n = 841
while n > 0:
    content = driver.page_source
    paldat = BeautifulSoup(content)  # , features="lxml")
    img = paldat.find_all("img")
    # Finds image url extension and formats to fit url
    for a in img:
        url = a.get("src")
        url = url.replace("../", "")
        if ".jpg" in url:
            # Retrieves image
            url = f"http://www.saps.org.uk/pollen/pollen/{url}"
            r = requests.get(url, allow_redirects=True)
            # Extracts species name from url to use as filename
            filename = url.split("/")[-1]
            filename = filename.replace("%20", " ")
            open(filename, "wb").write(r.content)
            n -= 1
        else:
            continue
    # Clicks link to next image
    driver.find_element_by_xpath("/html/body/center[2]/a").click()
