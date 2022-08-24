from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select

url = 'https://twiki.cern.ch/twiki/bin/view/AtlasPublic'

service = Service(ChromeDriverManager().install())

options = Options()
options.add_argument("--headless")
options.add_argument('--no-sandbox')
options.add_argument('--disable-gpu')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('disable-infobars')
options.add_argument("--disable-extensions")

with webdriver.Chrome(options=options, service=service) as driver:
    driver.get(url)
    # click to expand other keywords
    driver.find_element(By.CSS_SELECTOR, "#row_show_signature span").click()
    # select the likelihood available keyword
    driver.find_element(By.ID, "Analysischaracteristics_Likelihood@available").click()
    # show all publications
    Select(
        driver.find_element(By.CSS_SELECTOR, 'select[name="publications_length"]')
    ).select_by_visible_text('All')
    # get all publications visible / left in the table
    rows = driver.find_elements('css selector', '#publications > tbody > tr')
    # iterate and print information
    for index, row in enumerate(rows):
        elements = row.find_elements("css selector", "td")
        short_title = elements[0]
        links = elements[-1].find_elements("css selector", "a")
        hepdata = [
            link.get_property('href')
            for link in links
            if link.text.lower() == 'hepdata'
        ]
        print(
            f'({index: 3d}) {short_title.text}\n  - {hepdata[0] if hepdata else "<missing link>"}'
        )
