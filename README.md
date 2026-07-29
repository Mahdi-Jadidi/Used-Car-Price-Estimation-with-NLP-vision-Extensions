# Used Car Price Estimation with NLP and Vision Extensions

This project estimates used-car prices from engineered vehicle features. It includes the Phase 3 end-to-end training and inference automation, with MLflow experiment tracking.

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

## Phase 3: End-to-End Automation

`pipeline.py` provides two separate pipelines:

1. **Training pipeline** (`python pipeline.py train`) loads the Phase 2 engineered training set, tunes Ridge, Random Forest, and Gradient Boosting models using three-fold cross-validation, selects the lowest-CV-MAE model, evaluates it on the held-out test set, and saves the selected model.
2. **Prediction pipeline** (`python pipeline.py predict`) loads the saved model, runs inference on the held-out/new input data, and writes the predictions to SQLite.

Run both stages with one command:

```bash
python pipeline.py
```

Generated artifacts are deliberately ignored by Git:

- `artifacts/best_model.joblib` - selected trained model
- `artifacts/metrics.json` - model name, hyperparameters, CV result, MAE, RMSE, and R2
- `artifacts/predictions.db` - SQLite database; final outputs are in the `model_predictions` table
- `mlruns/` - MLflow local tracking store, including the run, parameters, metrics, and model

To inspect experiment tracking after a run:

```bash
mlflow ui --backend-store-uri ./mlruns
```

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
python -m pip install -r requirements.txt
python pipeline.py            # train, evaluate, and save predictions
python pipeline.py train      # training pipeline only
python pipeline.py predict    # inference pipeline only (uses saved model)
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
- executes the end-to-end Phase 3 pipeline on every push and pull request

## Docker

The repository also includes a `Dockerfile` for optional containerized execution.

## Notes

- The original raw dataset/database is stored through Git LFS; run `git lfs pull` after cloning if you need to execute the legacy Phase 2 preprocessing scripts.
- The Phase 3 pipeline is runnable from the tracked `data/train_data.csv` and `data/test_data.csv` files, so CI does not depend on downloading the LFS raw database.
- The SQLite schema and SQL extraction logic are embedded in the scripts.
- Some preprocessing and feature engineering steps rely on text matching and multilingual car-name normalization.
- The final prediction results are always written to the `model_predictions` table in `artifacts/predictions.db`.

## Troubleshooting

- `unable to open database file`
  - Make sure the dataset was fetched with Git LFS and the `datasets/` folder contains the real `.db` file.
- `ModuleNotFoundError`
  - Install requirements with `pip install -r requirements.txt`.
- Encoding issues on Windows
  - Set UTF-8 mode with `chcp 65001` and `PYTHONIOENCODING=utf-8`.
