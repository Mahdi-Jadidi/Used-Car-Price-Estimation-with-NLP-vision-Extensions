import requests
from bs4 import BeautifulSoup
import selenium
import jdatetime
import datetime
import os
import sqlite3
import hashlib
import json
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from bs4 import BeautifulSoup
import time
import sys
from config import *
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main_directory import get_main_dir

day_dollar_price = {}
day_dollar_price["1781555400"] = "153,500"
TABLE_scrap_cars = "scrab_cars_table"

def persian_to_timestamp(year, month, day):
    gregorian_date = jdatetime.date(year, month, day).togregorian()
    dt = datetime.datetime(gregorian_date.year, gregorian_date.month, gregorian_date.day)
    return int(dt.timestamp())


def box_parser(driver, box):
    model = box.find_all('p', class_="whitespace-pre-line font-medium text-sm text-secondary-1000")
    date = box.find_all('p', class_="whitespace-pre-line font-normal text-xs text-secondary-700")
    price = box.find_all('p', class_="whitespace-pre-line font-medium text-sm text-secondary-1000")
   
    if(len(model) < 4):
        print("Bad Input")
        return []
    car_data = {
        "model": model[0].get_text(strip=True) ,
        "trim": model[1].get_text(strip=True) ,
        "year": model[2].get_text(strip=True) ,
        "price": model[3].get_text(strip=True),
        "manual": date[0].get_text(strip=True),
        "kms_driven": date[1].get_text(strip=True),
        "Date": date[3].get_text(strip=True),
        "dollar_price": 0
    }
    year, month, day = car_data['Date'].split('/')
    year = int(year)
    month = int(month)
    day = int(day)
    time_stamp = persian_to_timestamp(year, month, day)
    print("go")
    if(not time_stamp in day_dollar_price):
        print("in")
        url = f"https://www.navasan.net/en/dayRates.php?item=usd_sell&date={time_stamp}&tz=Asia%2FTehran"
        driver.get(url)
        time.sleep(2)
        
        print(time_stamp)
        soup = BeautifulSoup(driver.page_source, "html.parser")
    #    dollar_price = soup.find_all('div', class_="idesc lastrate pos")
     #   if(len(dollar_price) == 0):
      #      dollar_price = soup.find_all('div', class_="idesc lastrate neg")
        print("GOT")
        dollar_price = soup.select('div.idesc.lastrate')

        dollar_price = dollar_price[0].get_text(strip=True).split()[0]
        dollar_price = dollar_price.split('.')[0]
        day_dollar_price[time_stamp] = dollar_price
        
        print("OUT")
    car_data["dollar_price"] = day_dollar_price[time_stamp]
    
    return car_data

def create_database():
    conn = sqlite3.connect(get_main_dir() + '/datasets/final_car_database.db')
    cursor = conn.cursor()
    
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_scrap_cars} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT,
            trim TEXT,
            year TEXT,
            price TEXT,
            transmission TEXT,
            kms_driven TEXT,
            listing_date TEXT,
            dollar_price TEXT,
            record_hash TEXT UNIQUE
        )
    """)
    
    conn.commit()
    return conn, cursor

def insert_car(cursor, car_data):
    record_hash = hashlib.sha256(json.dumps(car_data, sort_keys=True).encode()).hexdigest()
    
    cursor.execute(f"SELECT id FROM {TABLE_scrap_cars} WHERE record_hash = ?", (record_hash,))
    if cursor.fetchone():
        return None
    
    cursor.execute(f"""
        INSERT INTO {TABLE_scrap_cars} (model, trim, year, price, transmission, kms_driven, dollar_price, listing_date, record_hash)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        car_data.get('model', ''),
        car_data.get('trim', ''),
        car_data.get('year', ''),
        car_data.get('price', ''),
        car_data.get('manual', ''),
        car_data.get('kms_driven', ''),
        car_data.get('dollar_price', ''),
        car_data.get('Date', ''),
        record_hash
    ))
    
    return cursor.lastrowid


DRIVER_PATH = get_main_dir() + "/chromedriver-linux64/chromedriver"  # Change this to your actual path

service = Service(DRIVER_PATH)
options = webdriver.ChromeOptions()
options.add_argument("--headless")

driver = webdriver.Chrome(service=service, options=options)

car_loaded = 0
page_number = 120

conn, cursor = create_database()
cursor.execute(f"SELECT COUNT(*) FROM {TABLE_scrap_cars}")
count = cursor.fetchone()[0]
while count < 5000:
    print("car_")
    url = f"https://khodro45.com/auction-history/?tab=buy&page={page_number}"
    driver.get(url)
    time.sleep(3)
  
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(4)
    print("GOT data")
    soup = BeautifulSoup(driver.page_source, "html.parser")
    car_boxes = soup.find_all('div', class_="bg-common-100 border-secondary-50 mb-4 flex flex-col gap-0.5 overflow-hidden rounded-lg border")
    for box in car_boxes:
        car_data = box_parser(driver, box)
        print(car_data)
        insert_car(cursor, car_data)
    conn.commit()
    
    page_number += 1
    cursor.execute(f"SELECT COUNT(*) FROM {TABLE_scrap_cars}")
    count = cursor.fetchone()[0]
    print(f"Page {page_number} | Total cars: {count}")


driver.quit()