# 🚗 Used Car Price Estimation with NLP & Vision Extensions

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-orange)]()
[![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-blue)]()
[![SQLite](https://img.shields.io/badge/SQLite-Database-lightgrey)]()
[![GitHub Actions](https://img.shields.io/badge/GitHub-Actions-success)]()

An end-to-end **MLOps pipeline** for predicting **used car prices** using structured vehicle data enhanced with **Natural Language Processing (NLP)** and **Computer Vision** techniques. The project automates the complete machine learning lifecycle, from data preprocessing and feature extraction to model training, evaluation, experiment tracking, and prediction generation.

---

# ✨ Features

- 🚀 End-to-end automated machine learning pipeline
- 🧹 Data preprocessing and feature engineering
- 📝 NLP-based feature extraction
- 🖼️ Computer Vision feature integration
- 🤖 Multiple regression model training
- 🎯 Hyperparameter tuning
- 📊 Automatic best model selection
- 📈 MLflow experiment tracking
- 💾 SQLite database integration
- 🐳 Docker support
- ⚙️ GitHub Actions CI pipeline

---

# 🏗️ Project Architecture

```text
Dataset
   │
   ▼
Load Data
   │
   ▼
Preprocessing
   │
   ▼
Feature Engineering
   │
   ├───────────────┐
   │               │
   ▼               ▼
 NLP Features   Vision Features
   │               │
   └───────┬───────┘
           ▼
Merged Feature Matrix
           │
           ▼
Model Training
           │
           ▼
Hyperparameter Tuning
           │
           ▼
Best Model Selection
           │
           ▼
Evaluation
           │
           ▼
MLflow Logging
           │
           ▼
Predictions
           │
           ▼
SQLite Database
```

---

# 📂 Repository Structure

```text
.
├── datasets/
│   ├── car_data.csv
│   └── final_car_database.db
│
├── data/
│   ├── train_data.csv
│   └── test_data.csv
│
├── scripts/
│   ├── preprocess.py
│   ├── load_dataset.py
│   ├── database_connection.py
│   ├── make_predictions.py
│   ├── car_LangChain_feature_extractor.py
│   ├── model_training.py
│   └── ...
│
├── pipeline.py
├── main_directory.py
├── requirements.txt
├── Dockerfile
├── job.yaml
└── README.md
```

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- LangChain
- OpenAI API
- MLflow
- SQLite
- Docker
- GitHub Actions

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/Mahdi-Jadidi/Used-Car-Price-Estimation-with-NLP-vision-Extensions.git
cd Used-Car-Price-Estimation-with-NLP-vision-Extensions
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# 📦 Dataset

The project uses a structured dataset of used vehicles containing features such as:

- Brand
- Model
- Manufacturing Year
- Mileage
- Fuel Type
- Transmission
- Engine Specifications
- Selling Price

Additional semantic information is extracted using **Natural Language Processing (NLP)** and **Computer Vision** modules.

---

# ▶️ Running the Pipeline

Run the complete automated pipeline:

```bash
python pipeline.py
```

Alternatively, execute individual stages:

```bash
python pipeline.py train
```

```bash
python pipeline.py predict
```

---

# 🤖 Machine Learning Workflow

The pipeline performs the following stages automatically:

1. Load the dataset
2. Clean missing and inconsistent data
3. Perform feature engineering
4. Extract NLP features
5. Extract Vision features
6. Train multiple regression models
7. Tune hyperparameters
8. Evaluate model performance
9. Select the best-performing model
10. Log experiments with MLflow
11. Generate predictions
12. Store predictions in SQLite

---

# 📈 Experiment Tracking

The project integrates **MLflow** to track:

- Training parameters
- Performance metrics
- Model artifacts
- Best model versions
- Experiment history

Launch the MLflow UI locally:

```bash
mlflow ui --backend-store-uri ./mlruns
```

---

# 📊 Models Evaluated

Several regression algorithms are trained and compared, including:

- Linear Regression
- Ridge Regression
- Random Forest Regressor
- Gradient Boosting Regressor

The best-performing model is selected automatically based on evaluation metrics.

---

# 💾 Outputs

After execution, the pipeline generates:

```text
artifacts/
├── best_model.pkl
├── metrics.json
├── predictions.csv
├── prediction_database.db
└── mlruns/
```

---

# 🔄 Continuous Integration

GitHub Actions automatically:

- Install dependencies
- Run the preprocessing pipeline
- Train the model
- Validate generated artifacts
- Ensure reproducibility

---

# 🐳 Docker Support

Build the Docker image:

```bash
docker build -t used-car-price .
```

Run the container:

```bash
docker run used-car-price
```

---

# 🔮 Future Improvements

- Add XGBoost and LightGBM models
- Deploy using FastAPI
- Build an interactive Streamlit dashboard
- Register models with the MLflow Model Registry
- Continuous model retraining
- Data drift detection and monitoring
- Cloud deployment with Kubernetes

---

# 👨‍💻 Author




