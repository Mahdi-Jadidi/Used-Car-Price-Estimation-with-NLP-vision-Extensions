from scripts.load_data import load_scrap_data, load_original_data, load_car_names_data
import os
import sys
import difflib
import numpy as np
from scripts.config import *
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from main_directory import get_main_dir
machine_makers = [
    "Iran Khodro", "SAIPA", "Pars Khodro", "Bahman", "Kerman", "Modiran", 
    "Toyota", "Volkswagen Group", "Hyundai Motor Group", "General Motors", 
    "Ford", "Stellantis", "Honda", "BMW Group", "Mercedes-Benz Group", 
    "Tesla", "BYD", "Great Wall", "SAIC Motor", "Tata Motors",
    "Caterpillar", "Komatsu", "Hitachi Construction Machinery", "Volvo Group", 
    "Liebherr", "Terex", "SANY", "XCMG", "Zoomlion", "HD Hyundai", 
    "John Deere", "CNH Industrial", "JCB", "Kobelco", "Doosan Bobcat",
    "Siemens", "ABB", "Bosch Rexroth", "Mitsubishi Heavy Industries", 
    "FANUC", "KUKA", "Rockwell Automation", "Honeywell", "Emerson Electric", 
    "Schneider Electric", "Parker Hannifin", "Illinois Tool Works", 
    "Sandvik", "Atlas Copco", "DMG MORI", "Yamazaki Mazak", "Makino", 
    "Okuma", "JTEKT", "DN Solutions", "HAAS", "TRUMPF", "AMADA", 
    "HYUNDAI WIA", "Schuler", "Grob", "Chiron", "General Electric", 
    "Halliburton", "Applied Materials", "Kennametal", "Canon", "Tetra Laval",
    "Bajaj Auto", "TVS Motor Company", "Hero MotoCorp", "Royal Enfield", 
    "Yamaha Motor", "Suzuki Motorcycle", "Kawasaki Heavy Industries", 
    "Piaggio", "KTM", "Ducati", "Triumph", "Harley-Davidson", 
    "MV Agusta", "Aprilia", "Moto Guzzi", "Husqvarna", "GasGas", 
    "Benelli", "CFMoto", "Zontes", "QJMotor", "Voge", "Benda", 
    "Zero Motorcycles", "Energica", "LiveWire", "Ola Electric", 
    "Ather Energy", "Simple Energy", "Revolt Motors"
]
transmission_types = ["manual transmission", "automatic transmission", "دنده اتوماتیک", "دنده دستی"]
car_prices_usd = {
    "پژو_207": 10000,
    "پژو_206": 6367,
    "پژو_پارس": 7733,
    "دنا_پلاس": 16033,
    "پژو_206 SD": 17000,
    "تیبا_صندوق‌دار": 6160,
    "سمند_سورن": 4667,
    "ساینا_S": 8200,
    "تیبا_هاچ‌بک": 5933,
    "ام وی ام_X22": 22000,
    "کوییک_ساده": 7667,
    "سمند_LX": 7867,
    "شاهین_G": 10867,
    "کوییک_R": 10000,
    "رانا_پلاس": 10867,
    "کوییک_GX": 11000,
    "کوییک_GXR": 7667,
    "کوییک_S": 7933,
    "ساینا_EX": 8500,
    "کوییک_SR": 11000,
    "رنو_تندر90": 8350,
    "تارا_V4 اتوماتیک": 17000,
    "دنا_معمولی": 10000,
    "پژو_405": 12000,
    "دانگ فنگ_H30 کراس": 11667,
    "چری_آریزو 5T": 23000,
    "جک_S5": 24333,
    "لیفان_X60": 16000,
    "جک_J4": 16000,
    "هایما_S7": 24700,
    "برلیانس_H330": 12267,
    "ام وی ام_315 هاچ‌بک": 5933,
    "کی ام سی_J7": 31000,
    "ام وی ام_X33 کراس": 15633,
    "رنو_ساندرو استپ وی": 17500,
    "چری_تیگو 5": 16000,
    "دنا_پلاس EF7P": 25000,
    "ام وی ام_530": 5867,
    "اطلس_G": 9467,
    "ام وی ام_X55": 24987,
    "رانا_LX": 7000,
    "جیلی_امگرند 7": 11000,
    "چری_تیگو 7": 18933,
    "تارا_V1 پلاس": 13000,
    "کی ام سی_T8": 38667,
    "کیا_سراتو (مونتاژ)": 19667,
    "کوییک_اتوماتیک فول پلاس": 8333,
    "چانگان_CS35 (مونتاژ)": 21000,
    "برلیانس_H230": 13000,
    "رنو_ساندرو": 15933,
    "فونیکس_FX": 30000,
    "شاهین_GL": 12667,
    "برلیانس_H320": 9267,
    "رنو_پارس تندر": 10333,
    "فونیکس_آریزو 6 پرو": 28000,
    "تارا_V2": 14333,
    "جک_J5": 12000,
    "چانگان_CS55 پلاس": 26000,
    "بک (بی ای سی)_X3 Pro": 18667,
    "اطلس_GL": 8800,
    "ام وی ام_550": 7667,
    "فیدلیتی_پرایم": 30667,
    "فیدلیتی_الیت": 32333,
    "ام وی ام_X33": 8933,
    "دیگنیتی_پرایم": 29333,
    "فونیکس_تیگو 7 پرو": 32000,
    "چری_آریزو 5": 18857,
    "جک_S3": 18533,
    "کی ام سی_X5": 27800,
    "تارا_V1": 11593,
    "لاماری_ایما": 36000,
    "سهند_S": 9200,
    "فردا_SX5": 25333,
    "هیوندای_سانتافه (ix45)": 45000,
    "ساینا_GX": 8800,
    "هیوندای_النترا": 38000,
    "رنو_تلیسمان": 44000,
    "ام وی ام_110S": 5200,
    "فونیکس_آریزو 6 جی تی (Z6)": 30000,
    "کیا_اپتیما": 35333,
    "برلیانس_H220": 8333,
    "هایما_S5": 25667,
    "ریسپکت_پرایم": 23000,
    "رانا_EL": 6000,
    "لیفان_X50": 9600,
    "شاهین_پلاس اتوماتیک": 17333,
    "لیفان_820": 15667,
    "ام وی ام_X33s": 12333,
    "پژو_207 SD": 18000,
    "جیلی_GC6": 12000,
    "هوندا_سیتی": 37667,
    "مزدا_3 جدید صندوق‌دار (مونتاژ)": 46000,
    "تویوتا_کرولا": 36333,
    "میتسوبیشی_ASX": 33333,
    "کی ام سی_K7": 30000,
    "کی ام سی_ایگل": 16667,
    "هیوندای_سوناتا": 46000,
    "لیفان_520i": 3667,
    "میتسوبیشی_اوتلندر": 116000,
    "سوزوکی_فرانکس هیبرید": 38667,
    "دیگنیتی_پرستیژ": 35333,
    "سانگ یانگ (کی جی ام)_تیوولی": 36333,
    "فیدلیتی_پرستیژ": 39333,
    "دایون_Y5": 21533,
    "جتا_VS5": 33333,
    "برلیانس_C3 کراس": 13000,
    "چانگان_CS35 پلاس": 24000,
    "ساینا_پلاس": 6383,
    "رنو_فلوئنس": 23000,
    "چری_آریزو 6": 23333,
    "نیسان_جوک": 32667,
    "پژو_2008": 25867,
    "هیوندای_توسان (ix35)": 32900,
    "تویوتا_کرولا کراس هیبرید": 44467,
    "مکث موتور_کلوت": 23667,
    "کی ام سی_T9": 38667,
    "سانگ یانگ (کی جی ام)_کوراندو": 39667,
    "اطلس_S": 5107,
    "هایما_8S": 32000,
    "رنو_کپچر": 26000,
    "اس دبلیو ام_G01": 22000,
    "لیفان_620": 5933,
    "کیا_اسپورتیج": 49667,
    "هایما_S5 پرو": 26600,
    "ام جی_550": 12000,
    "بسترن_B30": 14667,
    "ری را_EF7P": 23333,
    "لوکانو_L7": 43000,
    "رنو_سیمبل": 14667,
    "ام جی_GS": 25667,
    "آلفارومئو_جولیتا": 33333,
    "تویوتا_لوین": 28000,
    "فونیکس_تیگو 8 پرو مکس (F8 Max)": 35000,
    "ام وی ام_315 صندوق‌دار": 6000
}
def get_kms_driven(x):
    x = x.split(' ')[0]
    return convert_to_int(x)
def make_year_correct(x):
    x = int(x)
    if(x < 1500):
        x += 621
    return x
def most_similar(candidates, x):
    x = str(x).strip().casefold()
    if not x:
        return candidates[0]

    def score(candidate):
        candidate_norm = str(candidate).strip().casefold()
        if candidate_norm in x or x in candidate_norm:
            return 1.0
        return difflib.SequenceMatcher(None, candidate_norm, x).ratio()

    return max(candidates, key=score)
def convert_to_int(x):
    try:
        return int(x.replace(',', '').split('-')[0])
    except Exception:
        print(x)
        return None
def most_similar_transmission(x):
    return most_similar(transmission_types, x)
def clean_df_scrap(df):
    print('go')
    df['price'] = df['price'].apply(convert_to_int)
    df['dollar_price'] = df['dollar_price'].apply(convert_to_int)
    df['price'] /= df['dollar_price']
    df['transmission'] = df['transmission'].apply(most_similar_transmission).map({"manual transmission":"Manual", "automatic transmission": "Automatic",
                                                                                  "دنده اتوماتیک": "Automatic", 
                                                                                  "دنده دستی": "Manual"})
    df[COL_Car_Name] = df['model'].astype(str) + '_' + df['trim'].astype(str)
    df = df.drop(columns=['model', 'trim'])
    df['year'] = df['year'].apply(make_year_correct)
    df['kms_driven'] = df['kms_driven'].apply(get_kms_driven)
    df.rename(columns={'year': COL_Year, 'price':COL_Sel_Price, 'transmission': COL_Trans, 'kms_driven':COL_Kms_Driven, 'Car_Name':COL_Car_Name}, inplace=True)
    df[COL_Fuel_Type] = 'Petrol'
    df[COL_Sell_Type] = 'Dealer'
    df[COL_Owner] = 0
    df[COL_Sel_Price] /= 100
    df[COL_Pres_Price] = df[COL_Car_Name].map(car_prices_usd) / 1000 
    df = df.drop(columns=['record_hash', 'dollar_price', 'listing_date', 'id'])
    print(sum(df[COL_Sel_Price] <= df[COL_Pres_Price]))
    print("_________________________")
    print(df.head(30))
  #  df['model'] = df['model'].apply(translate_text)
    return df

def clean_concat_df(df):
    mask = df[COL_Sel_Price] <= df[COL_Pres_Price]
    df = df[mask]
    df.dropna()
    return df
def translate_text(text, source_lang='fa', target_lang='en'):
    return str(text)

df_scrap = load_scrap_data()
df_scrap_clean = clean_df_scrap(df_scrap)

df_original = load_original_data()
result = pd.concat([df_scrap_clean, df_original], axis=0, ignore_index=True)

df_concat_clean = clean_concat_df(result)
print(df_concat_clean.info())
print(df_concat_clean.head())
