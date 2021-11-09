from selenium import webdriver

url = 'https://twiki.cern.ch/twiki/bin/view/AtlasPublic'

with webdriver.Chrome() as driver:
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
