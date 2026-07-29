import pandas as pd
from scripts.database_connection import get_connection
import sys
import os
from scripts.config import *
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main_directory import get_main_dir
def load_scrap_data():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM scrab_cars_table", conn)
    conn.close()
    return df
def load_original_data():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM car_original", conn)
    conn.close()
    return df
def load_car_names_data():
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM car_name_info", conn)
    conn.close()
    return df

df_scrap = load_original_data()
print(df_scrap.head(40)) 