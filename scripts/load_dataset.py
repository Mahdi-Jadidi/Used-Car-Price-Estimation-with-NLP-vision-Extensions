import pandas as pd
import sqlite3
import sys
import os
from config import *
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main_directory import get_main_dir
database_path = get_main_dir() + "/datasets/car_data.csv"
df = pd.read_csv(database_path)
conn = sqlite3.connect(get_main_dir() + '/datasets/final_car_database.db')
cursor = conn.cursor()

TABLE_car_name_info = "car_name_info"
TABLE_cars_original = "car_original"
COL_Car_Name = "Car_Name"
COL_Year = "Year"
COL_Sel_Price = "Selling_Price"
COL_Pres_Price = "Present_Price"
COL_Kms_Driven = "Kms_Driven"
COL_Fuel_Type = "Fuel_Type"
COL_Sell_Type = "Seller_Type"
COL_Trans = "Transmission"
COL_Owner = "Owner"


COL_Car_Brand = "car_brand"
COL_Img_URL = "car_image_url"

cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {TABLE_cars_original} (
        car_id INTEGER PRIMARY KEY AUTOINCREMENT,
        {COL_Car_Name} TEXT,
        {COL_Year} INTEGER,
        {COL_Sel_Price} REAL,
        {COL_Pres_Price} REAL,
        {COL_Kms_Driven} INTEGER,
        {COL_Fuel_Type} TEXT,
        {COL_Sell_Type} TEXT,
        {COL_Trans} TEXT,
        {COL_Owner} INTEGER
    );
""")

cursor.execute(f"""
    CREATE TABLE IF NOT EXISTS {TABLE_car_name_info} (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        {COL_Car_Name} TEXT ,
        {COL_Year} INTEGER,
        {COL_Car_Brand} INTEGER,
        {COL_Img_URL} TEXT
    );
""")
conn.commit()

df.head()

df.to_sql(TABLE_cars_original, conn, if_exists="replace", index=False)

unique_cars = df.drop_duplicates(subset=[COL_Car_Name, COL_Year])
unique_cars = unique_cars.filter(items=[COL_Car_Name, COL_Year])

unique_cars.to_sql(TABLE_car_name_info, conn, if_exists='replace', index=False)