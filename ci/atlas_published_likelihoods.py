from selenium import webdriver
from selenium.webdriver.chrome.options import Options

url = 'https://twiki.cern.ch/twiki/bin/view/AtlasPublic'

options = Options()
options.add_argument("--headless")
options.add_argument('--no-sandbox')
options.add_argument('--disable-gpu')
options.add_argument('--disable-dev-shm-usage')
options.add_argument('disable-infobars')
options.add_argument("--disable-extensions")

with webdriver.Chrome(
    chrome_options=options, executable_path='/usr/bin/chromedriver'
) as driver:
    driver.get(url)
    driver.execute_script("addKeyword('Analysischaracteristics_Likelihood@available');")
    rows = driver.find_elements('css selector', '#paperListTbody tr')

    for row in rows:
        elements = row.find_elements("css selector", "td")
        short_title = elements[0]
        links = elements[-1].find_elements("css selector", "a")
        hepdata = [
            link.get_property('href')
            for link in links
            if link.text.lower() == 'hepdata'
        ]
        print(f'{short_title.text}\n  - {hepdata[0] if hepdata else "<missing link>"}')
