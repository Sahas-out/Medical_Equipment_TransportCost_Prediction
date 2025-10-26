import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def load_preprocessed_data(train_path):
    df = pd.read_csv(train_path)
    # Use log target if available, else raw
    if 'Transport_Cost_Log' in df.columns:
        y = df['Transport_Cost_Log']
        use_log = True
        shift = df['Target_Shift_Value'].iloc[0] if 'Target_Shift_Value' in df.columns else 0
    else:
        y = df['Transport_Cost']
        use_log = False
        shift = 0
    X = df.drop(['Transport_Cost', 'Transport_Cost_Log', 'Target_Shift_Value'], axis=1, errors='ignore')
    # Remove any non-feature columns (like Hospital_Id if present)
    if 'Hospital_Id' in X.columns:
        X = X.drop('Hospital_Id', axis=1)
    return X, y, use_log, shift

# Load data using the preprocessing function
X_train_full, y_train_full, use_log, shift = load_preprocessed_data('data/train_processed.csv')
test_data_orig = pd.read_csv('data/test_processed.csv')

# Work on copies to preserve original processed data
test_data = test_data_orig.copy()

# Split training data into train/validation
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=42)

# Test data (no target column - this is what we predict)
X_test = test_data
if 'Hospital_Id' in X_test.columns:
    X_test = X_test.drop('Hospital_Id', axis=1)

# Random Forest with hyperparameter tuning
rf_model = RandomForestRegressor(random_state=42, n_jobs=-1)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt']
}

grid_search = GridSearchCV(rf_model, param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")

# Validate with best model
best_rf = grid_search.best_estimator_
val_pred = best_rf.predict(X_val)
val_mse = mean_squared_error(y_val, val_pred)
val_r2 = r2_score(y_val, val_pred)

print(f"Validation MSE: {val_mse:.4f}")
print(f"Validation R²: {val_r2:.4f}")

# Final predictions on test data
test_pred = best_rf.predict(X_test)

# Apply inverse transformation if log target was used
if use_log:
    test_pred = np.expm1(test_pred) - shift
    test_pred = np.maximum(test_pred, 0)

# Save predictions to file
predictions_df = pd.DataFrame({
    'Hospital_Id': test_data['Hospital_Id'],
    'Predicted_Transport_Cost': test_pred
})
predictions_df.to_csv('output/random_forest_predictions.csv', index=False)
print("Predictions saved to output/random_forest_predictions.csv")
