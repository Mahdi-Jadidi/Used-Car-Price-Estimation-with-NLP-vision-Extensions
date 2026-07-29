import sqlite3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main_directory import get_main_dir
def get_connection():
    DB_PATH = get_main_dir() + "/datasets/final_car_database.db"
    print(DB_PATH)
    return sqlite3.connect(DB_PATH)
print(get_connection())