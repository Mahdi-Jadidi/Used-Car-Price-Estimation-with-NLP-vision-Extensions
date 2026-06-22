markdown
# Car Price Prediction - Data Science Pipeline

A modular data science pipeline for loading, cleaning, and feature-engineering
car listing data from a SQLite database.

## Project Structure

.

├── database_connection.py # SQLite connection handler (get_connection)

├── config.py # Project configuration and constants

├── main_directory.py # Path / directory management

├── pipeline.py # Main entry point that runs all stages

├── scripts/

│ ├── load_data.py # Loads data from database tables

│ ├── preprocess.py # Cleans, translates, and embeds text data

│ └── feature_enginering.py # Splits data and transforms features

├── datasets/ # Database files

└── data/output/ # Pipeline outputs
Requirements

    Python 3.8+
    Dependencies in requirements.txt

Installation

bash

pip install -r requirements.txt

On Windows, set UTF-8 encoding to avoid Unicode errors:

powershell

chcp 65001

$env:PYTHONIOENCODING=“utf-8”
Usage

bash

python pipeline.py
Pipeline Stages
1. Load Data (load_data.py)

Reads from three database tables:

    scrab_cars_table via load_scrap_data()
    car_original via load_original_data()
    car_name_info via load_car_names_data()

2. Preprocess (preprocess.py)

    Cleans scraped data (clean_df_scrap)
    Parses kilometers, fixes years, normalizes transmission values
    Translates text with deep-translator (GoogleTranslator)
    Generates text embeddings with sentence-transformers
    Concatenates scraped + original data into df_concat_clean

3. Feature Engineering (feature_enginering.py)

    Brand matching via most_similar_brand
    80/20 train/test split
    StandardScaler for numeric features
    OneHotEncoder for categorical features
    Produces df_train_transformed and df_test_transformed

4. Save Outputs

Writes results to data/output/:

    cleaned_data.csv
    train_data.csv
    test_data.csv
    pipeline_data.pkl

Key Dependencies
Library 	Purpose
pandas / numpy 	Data processing
scikit-learn 	Feature transformation & split
sentence-transformers 	Text embeddings
deep-translator 	Text translation
requests 	HTTP requests
Common Errors

    ModuleNotFoundError: run pip install -r requirements.txt
    UnicodeEncodeError (Windows): set chcp 65001 and PYTHONIOENCODING=utf-8
    Database connection error: confirm the DB file exists under datasets/ and thepath in config.py is correct