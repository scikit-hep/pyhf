#!/usr/bin/env python

import argparse
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait


def main(args):
    driver = webdriver.Firefox()
    driver.get(args.url)
    WebDriverWait(driver, 10)
    driver.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', dest='url',
                        type=str, default=None, help='URL for Selinium to open')
    args = parser.parse_args()

    main(args)
