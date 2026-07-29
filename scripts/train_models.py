import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_and_split_data():
    train_df = pd.read_csv('data/train_data.csv')
    test_df = pd.read_csv('data/test_data.csv')
    
    X = train_df.drop('price', axis=1)
    y = train_df['price']
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_test_final = test_df.drop('price', axis=1)
    y_test_final = test_df['price']
    
    return X_train, X_val, y_train, y_val, X_test_final, y_test_final

def train_and_evaluate():
    X_train, X_val, y_train, y_val, X_test_final, y_test_final = load_and_split_data()

    models = {
        "Ridge_Regression": {
            "model": Ridge(),
            "params": {
                'alpha': [0.1, 1.0, 10.0, 100.0] 
            }
        },
        "Random_Forest": {
            "model": RandomForestRegressor(random_state=42),
            "params": {
                'n_estimators': [50, 100],
                'max_depth': [10, 20], 
                'min_samples_split': [2, 5]
            }
        },
        "Gradient_Boosting": {
            "model": GradientBoostingRegressor(random_state=42),
            "params": {
                'n_estimators': [50, 100],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5]
            }
        }
    }

    best_models = {}

    for model_name, config in models.items():
        print(f"\nTraining and Tuning {model_name}...")
        
        grid_search = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            cv=3, 
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            verbose=2
        )
        
        grid_search.fit(X_train, y_train)
        best_models[model_name] = grid_search.best_estimator_
        
        print(f"Best parameters for {model_name}: {grid_search.best_params_}")

    print("\n--- Final Evaluation on Unseen Test Data ---")
    
    overall_best_model = None
    best_r2 = -float('inf')
    best_model_name = ""

    for model_name, model in best_models.items():
        y_pred = model.predict(X_test_final)
        
        mae = mean_absolute_error(y_test_final, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test_final, y_pred))
        r2 = r2_score(y_test_final, y_pred)
        
        print(f"Results for {model_name}:")
        print(f"  MAE (Mean Absolute Error): {mae:.2f}")
        print(f"  RMSE (Root Mean Squared Error): {rmse:.2f}") 
        print(f"  R^2 (R-squared): {r2:.4f}\n")

        if r2 > best_r2:
            best_r2 = r2
            overall_best_model = model
            best_model_name = model_name

    print("--- Model Persistence ---")
    model_filename = 'best_model.pkl'
    joblib.dump(overall_best_model, model_filename)
    print(f" Best model ({best_model_name}) successfully saved to '{model_filename}' with R^2: {best_r2:.4f}")

if __name__ == "__main__":
    train_and_evaluate()
