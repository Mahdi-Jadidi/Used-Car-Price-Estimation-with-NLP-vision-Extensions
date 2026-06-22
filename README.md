# Used Car Price Estimation with NLP and Vision Extensions

This project builds a data science pipeline for used-car price estimation using a SQLite-backed dataset, text-aware preprocessing, and feature engineering for modeling preparation.

The repository is structured to match the Phase 2 assignment requirements:
- database storage
- preprocessing
- feature engineering
- automated pipeline execution
- GitHub Actions CI
- optional Docker support

## Dataset And Git LFS

The dataset is too large for a normal GitHub upload, so it is tracked with Git LFS.

Tracked dataset files:
- `datasets/car_data.csv`
- `datasets/final_car_database.db`

If you clone the repository locally, run:

```bash
git lfs install
git lfs pull
```

This restores the real dataset files instead of the small pointer files stored in the Git commit history.

## Project Structure

```text
.
├── pipeline.py
├── main_directory.py
├── requirements.txt
├── Dockerfile
├── job.yaml
├── datasets/
│   ├── car_data.csv
│   └── final_car_database.db
├── scripts/
│   ├── database_connection.py
│   ├── load_data.py
│   ├── preprocess.py
│   ├── feature_enginering.py
│   ├── load_dataset.py
│   ├── NetScrapping.py
│   ├── save_to_database.py
│   └── config.py
└── .github/workflows/pipeline.yml
```

## What The Pipeline Does

`pipeline.py` runs the project end to end:

1. Connects to the SQLite database.
2. Loads raw tables into pandas DataFrames.
3. Cleans and preprocesses the scraped/original car data.
4. Builds engineered features and splits train/test sets.
5. Saves pipeline outputs to disk.

Generated outputs are written to:
- `data/output/cleaned_data.csv`
- `data/output/train_data.csv`
- `data/output/test_data.csv`
- `data/output/pipeline_data.pkl`

## Main Scripts

- `scripts/database_connection.py` - opens the SQLite database connection.
- `scripts/load_data.py` - loads tables from the database.
- `scripts/preprocess.py` - cleans and normalizes the raw data.
- `scripts/feature_enginering.py` - creates brand features and prepares train/test matrices.

## Requirements

- Python 3.8 or newer
- Dependencies listed in `requirements.txt`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run The Pipeline

From the repository root:

```bash
python pipeline.py
```

On Windows, if you run into console encoding problems:

```powershell
chcp 65001
$env:PYTHONIOENCODING="utf-8"
python pipeline.py
```

## CI/CD

The repository includes GitHub Actions in `.github/workflows/pipeline.yml`.

The workflow:
- runs on push and pull request events to `main`
- installs dependencies from `requirements.txt`
- executes `python pipeline.py`

## Docker

The repository also includes a `Dockerfile` for optional containerized execution.

## Notes

- The database file must exist in `datasets/final_car_database.db`.
- The SQLite schema and SQL extraction logic are embedded in the scripts.
- Some preprocessing and feature engineering steps rely on text matching and multilingual car-name normalization.
- The project is focused on the data pipeline and modeling preparation phase, not on final model training.

## Troubleshooting

- `unable to open database file`
  - Make sure the dataset was fetched with Git LFS and the `datasets/` folder contains the real `.db` file.
- `ModuleNotFoundError`
  - Install requirements with `pip install -r requirements.txt`.
- Encoding issues on Windows
  - Set UTF-8 mode with `chcp 65001` and `PYTHONIOENCODING=utf-8`.
