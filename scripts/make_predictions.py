import pandas as pd
import numpy as np
import joblib

def load_artifacts():
    return joblib.load('best_model.pkl')


def predict_with_confidence(model, X_new, fallback_rmse=2.00):
    
    if 'diff_year' in X_new.columns:
        X_new = X_new.rename(columns={'diff_year': 'Year_difference'})
        
    X_processed = pd.get_dummies(X_new)
    
    expected_features = model.feature_names_in_
    X_processed = X_processed.reindex(columns=expected_features, fill_value=0)
    
    mean_preds = model.predict(X_processed)
    
    if type(model).__name__ == "RandomForestRegressor":
        tree_preds = np.array([tree.predict(X_processed) for tree in model.estimators_])
        std_dev = np.std(tree_preds, axis=0)
        margin_of_error = 1.96 * std_dev 
        method = "Random Forest Variance"
    else:
        margin_of_error = np.full(len(mean_preds), 1.96 * fallback_rmse)
        method = f"Static RMSE (for {type(model).__name__})"
        
    return mean_preds, margin_of_error, method



if __name__ == "__main__":
    print("Loading best model...")
    try:
        pipeline = load_artifacts()
    except FileNotFoundError:
        print("Error: 'best_model.pkl' not found. Run train_models.py first.")
        exit()
        
    new_samples = pd.DataFrame({
        'Present_Price': [5.59, 9.54, 1.20],
        'Kms_Driven': [27000, 43000, 15000],
        'Fuel_Type': ['Petrol', 'Diesel', 'Petrol'],
        'Seller_Type': ['Dealer', 'Dealer', 'Individual'],
        'Transmission': ['Manual', 'Manual', 'Automatic'],
        'Owner': [0, 0, 1],
        'diff_year': [8, 9, 4] 
    })
    
    print("\nPredicting prices with 95% Confidence Interval...")
    
    prices, margins, method = predict_with_confidence(pipeline, new_samples, fallback_rmse=2.00)
    
    print(f"Uncertainty Method Used: {method}\n")
    print("-" * 40)
    
    for i in range(len(prices)):
        print(f"   Car {i+1}:")
        print(f"   Estimated Price: {prices[i]:.2f}")
        print(f"   Uncertainty Margin: ± {margins[i]:.2f} (95% CI)")
        
        lower_bound = max(0, prices[i] - margins[i]) 
        upper_bound = prices[i] + margins[i]
        
        print(f"   Expected Range: [{lower_bound:.2f}  to  {upper_bound:.2f}]\n")
