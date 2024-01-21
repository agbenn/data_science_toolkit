import json
import pandas as pd
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException, StaleElementReferenceException
from selenium import webdriver
import os
from bs4 import BeautifulSoup
import time
from selenium import webdriver
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options

from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.action_chains import ActionChains

import re

# Go get geckodriver from : https://github.com/mozilla/geckodriver/releases


def click_on_coordinates(driver, element, x, y):
    actions = ActionChains(driver)
    actions.move_to_element_with_offset(element, x, y)
    actions.click()
    actions.perform()
def ffx_preferences(dfolder, download=False):
    '''
    Sets the preferences of the firefox browser: download path.
    '''
    profile = webdriver.FirefoxProfile()
    # set download folder:
    profile.set_preference("browser.download.dir", dfolder)
    profile.set_preference("browser.download.folderList", 2)
    profile.set_preference("browser.download.manager.showWhenStarting", False)
    profile.set_preference("browser.helperApps.neverAsk.saveToDisk",
                           "application/msword,application/rtf, application/csv,text/csv,image/png ,image/jpeg, application/pdf, text/html,text/plain,application/octet-stream")
    
    #profile.add_extension('/Users/luisignaciomenendezgarcia/Dropbox/CLASSES/class_bse_text_mining/class_scraping_bse/booking/booking/ublock_origin-1.55.0.xpi')


    # this allows to download pdfs automatically
    if download:
        profile.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/pdf,application/x-pdf")
        profile.set_preference("pdfjs.disabled", True)

    options = Options()
    options.profile = profile
    return options


def start_up(link, dfolder, geko_path,donwload=True):
    # geko_path='/Users/luisignaciomenendezgarcia/Dropbox/CLASSES/class_bse_text_mining/class_scraping_bse/booking/geckodriver'
    # download_path='./downloads'
    os.makedirs(dfolder, exist_ok=True)

    options = ffx_preferences(dfolder,donwload)
    service = Service(geko_path)
    browser = webdriver.Firefox(service=service, options=options)
    # Enter the website address here
    browser.get(link)
    time.sleep(5)  # Adjust sleep time as needed
    return browser


def check_and_click(browser, xpath, type):
    '''
    Function that checks whether the object is clickable and, if so, clicks on
    it. If not, waits one second and tries again.
    '''
    ck = False
    ss = 0
    while ck == False:
        ck = check_obscures(browser, xpath, type)
        time.sleep(1)
        ss += 1
        if ss == 15:
            # warn_sound()
            # return NoSuchElementException
            ck = True
            # browser.quit()

def check_obscures(browser, xpath, type):
    '''
    Function that checks whether the object is being "obscured" by any element so
    that it is not clickable. Important: if True, the object is going to be clicked!
    '''
    try:
        if type == "xpath":
            browser.find_element('xpath',xpath).click()
        elif type == "id":
            browser.find_element('id',xpath).click()
        elif type == "css":
            browser.find_element('css selector',xpath).click()
        elif type == "class":
            browser.find_element('class name',xpath).click()
        elif type == "link":
            browser.find_element('link text',xpath).click()
    except (ElementClickInterceptedException) as e:
        print(e)
        error_string = str(e)
        intercepting_element = re.findall('another element (<.+>)', error_string)[0]
        return intercepting_element
    except (NoSuchElementException, StaleElementReferenceException):
        return False
    return True

def switch_to_iframe_click_element_switch_back(driver, iframe_locator, element_locator):
    # Switch to the iframe
    iframe = WebDriverWait(driver, 10).until(EC.frame_to_be_available_and_switch_to_it(iframe_locator))
    
    # Click the element within the iframe
    element = WebDriverWait(iframe, 10).until(EC.element_to_be_clickable(element_locator))
    element.click()
    
    # Switch back to the default content
    driver.switch_to.default_content()