"""
Pipeline stages:
1. Database connection
2. Data loading from SQLite
3. Data preprocessing and cleaning
4. Feature engineering and modeling prep
5. Save final outputs
"""

import os
import sys
import pandas as pd
import pickle
from datetime import datetime

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

def main():
    """Execute complete data pipeline."""
    
    print("="*60)
    print("CAR PRICE PREDICTION - DATA PIPELINE")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    print("[1/5] Testing database connection...")
    try:
        from scripts.database_connection import get_connection
        conn = get_connection()
        print("      [OK] Database connected successfully")
        conn.close()
    except Exception as e:
        print(f"      [ERROR] Database connection failed: {e}")
        return
    
    print("[2/5] Loading data from database...")
    try:
        from scripts.load_data import load_scrap_data, load_original_data, load_car_names_data
        
        df_scrap = load_scrap_data()
        df_original = load_original_data()
        df_names = load_car_names_data()
        
        print(f"      [OK] Scrap data: {len(df_scrap)} rows")
        print(f"      [OK] Original data: {len(df_original)} rows")
        print(f"      [OK] Car names: {len(df_names)} rows")
    except Exception as e:
        print(f"      [ERROR] Data loading failed: {e}")
        return
    
    print("[3/5] Preprocessing and cleaning data...")
    try:
        from scripts.preprocess import df_concat_clean
        
        print(f"      [OK] Cleaned data: {len(df_concat_clean)} rows, {len(df_concat_clean.columns)} columns")
        print(f"      [OK] Columns: {list(df_concat_clean.columns)}")
    except Exception as e:
        print(f"      [ERROR] Preprocessing failed: {e}")
        return
    
    print("[4/5] Feature engineering and model preparation...")
    try:
        from scripts.feature_enginering import df_train_transformed, df_test_transformed
        
        print(f"      [OK] Train set: {len(df_train_transformed)} rows, {len(df_train_transformed.columns)} features")
        print(f"      [OK] Test set: {len(df_test_transformed)} rows, {len(df_test_transformed.columns)} features")
    except Exception as e:
        print(f"      [ERROR] Feature engineering failed: {e}")
        return
    
    print("[5/5] Saving pipeline outputs...")
    try:
        output_dir = os.path.join(project_root, "data", "output")
        os.makedirs(output_dir, exist_ok=True)
        
        cleaned_path = os.path.join(output_dir, "cleaned_data.csv")
        df_concat_clean.to_csv(cleaned_path, index=False, encoding='utf-8-sig')
        print(f"      [OK] Cleaned data saved: {cleaned_path}")
        
        train_path = os.path.join(output_dir, "train_data.csv")
        df_train_transformed.to_csv(train_path, index=False, encoding='utf-8-sig')
        print(f"      [OK] Train data saved: {train_path}")
        
        test_path = os.path.join(output_dir, "test_data.csv")
        df_test_transformed.to_csv(test_path, index=False, encoding='utf-8-sig')
        print(f"      [OK] Test data saved: {test_path}")
        
        pickle_path = os.path.join(output_dir, "pipeline_data.pkl")
        with open(pickle_path, 'wb') as f:
            pickle.dump({
                'cleaned': df_concat_clean,
                'train': df_train_transformed,
                'test': df_test_transformed
            }, f)
        print(f"      [OK] Pickle file saved: {pickle_path}")
        
    except Exception as e:
        print(f"      [ERROR] Saving outputs failed: {e}")
        return
    
    print()
    print("="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {output_dir}")
    print()

if __name__ == "__main__":
    main()
