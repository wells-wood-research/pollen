import csv
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
import io
from PIL import Image
# Define path and driver location. Employ BeautifulSoup library and define parameters
cd = Path("coding")
driver = webdriver.Chrome(
    executable_path=r"C:/Users/pinto/OneDrive - University of Edinburgh/coding/chromedriver"
)
# Direct to example PalDat page to extract the names of each descriptive category to create headers for the outpput .csv
driver.get("https://www.paldat.org/pub/Abelia_grandiflora/305286")
content = driver.page_source
paldat = BeautifulSoup(content, features="lxml")
catlist = ["Genus", "Species"]
for c in paldat.find_all("span", class_="diag-label"):
    ca = c.extract()
    cat = ca.string.extract()
    catlist.append(cat)
with open("species.csv", "w", newline="") as out:
    out = csv.DictWriter(out, catlist)
    out.writeheader()
#Define URL dictionary
images_url = {}
# Loop directs to each page of genera by letter
for L in ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","W","X","Y","Z"]:
    driver.get(f"https://www.paldat.org/search/{L}")
    # Find all genera's URL appendages on the page by letter and add to list
    genus = driver.find_elements_by_class_name("genus")
    glist = []
    for a in genus:
        glist.append(str(a.text))
    for g in glist:
        # Direct to each genus page
        driver.get(f"https://www.paldat.org/search/genus/{g}")
        # The 'soup' has to be redefined on every page
        content = driver.page_source
        paldat = BeautifulSoup(content, features="lxml")
        # Find image URLs for all LM images for each species and add to image url dictionary
        for a in paldat.find_all("td", attrs={"data-grid-property": "LM"}):
            for b in a.find_all("a"):
                name = b.get("data-title")
                images_url.setdefault(name, [])
                url = b.get("href")
                images_url[name].append(url)
        else:
            driver.back()
        # Find all species URL appendage in the genus's page
        species = driver.find_elements_by_class_name("species")
        slist = []
        for a in species:
            slist.append(str(a.text).replace(" ", "_"))
        for s in slist:
            # Direct to each species in genus's page
            driver.get(f"https://www.paldat.org/pub/{s}")
            datlist = []
            content = driver.page_source
            paldat = BeautifulSoup(content, features="lxml")
            # Extract species name from page
            for c in paldat.find_all("h1", class_="species"):
                d = c.extract()
                name = d.string.extract()
                name = name.split()
                datlist.extend(name)
            tax_list = []
            for t in paldat.find("p"):
                tax_list.append(t)
            taxonomy = tax_list[1].replace("\n", "").split(",")
            # Extract all descriptive values for pollen species
            for v in paldat.find_all("span", class_="diag-value"):
                va = v.extract()
                val = va.string.extract()
                datlist.append(val)
            datlist.append(taxonomy)
            # Append list of species name, values and taxonomic description to .csv file
            with open("species.csv", "a", newline="") as out:
                out = csv.writer(out, catlist)
                out.writerow(datlist)

# Makes directory for images to be stored in
if not Path("paldat_images").exists:
    Path("paldat_images").mkdir()
# Loop to access each image from its URL and download it, naming it by the species name
for key, values in images_url.items():
    for val in values:
        image = requests.get(f"https://www.paldat.org{val}").content
        image_file = io.BytesIO(image)
        image = Image.open(image_file).convert('RGB')
        path = Path("paldat_images", f"{key}")
        image.save(path,'JPEG', quality = 95)
        image.close()