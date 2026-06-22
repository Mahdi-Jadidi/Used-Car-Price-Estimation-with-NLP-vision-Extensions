import sqlite3
import pandas as pd
from feature_enginering import df_train_transformed, df_test_transformed, y_train, y_test
from database_connection import get_connection
y_train_reset = y_train.reset_index(drop=True)
y_test_reset = y_test.reset_index(drop=True)

df_train_full = pd.concat([df_train_transformed, y_train_reset], axis=1)
df_test_full = pd.concat([df_test_transformed, y_test_reset], axis=1)

conn = get_connection()
df_train_full.to_sql('train_data', conn, if_exists='replace', index=False)
df_test_full.to_sql('test_data', conn, if_exists='replace', index=False)
conn.close()

print("Tables 'train_data' and 'test_data' saved to final_car_database.db")