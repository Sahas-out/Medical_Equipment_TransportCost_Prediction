import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
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

# Polynomial regression with hyperparameter tuning
# Create pipeline: PolynomialFeatures -> StandardScaler -> LinearRegression
poly_pipeline = Pipeline([
    ('poly', PolynomialFeatures()),
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

# Hyperparameter grid - optimized for speed and memory
param_grid = {
    'poly__degree': [1, 2],  # Reduced from [1, 2, 3] to avoid memory explosion
    'poly__interaction_only': [True],  # Only interactions, no x^2 terms
    'poly__include_bias': [True]  # Keep bias for better performance
}

grid_search = GridSearchCV(
    poly_pipeline, param_grid, 
    cv=3, scoring='r2', n_jobs=-1,  # Reduced CV from 5 to 3 for speed
    return_train_score=True
)
grid_search.fit(X_train, y_train)

print(f"Best params: {grid_search.best_params_}")
print(f"Best CV Score: {grid_search.best_score_:.4f}")

# Check for overfitting
train_scores = grid_search.cv_results_['mean_train_score']
val_scores = grid_search.cv_results_['mean_test_score']
best_idx = grid_search.best_index_
print(f"Train Score: {train_scores[best_idx]:.4f}")
print(f"Validation gap: {abs(train_scores[best_idx] - val_scores[best_idx]):.4f}")

# Validate with best model
best_poly = grid_search.best_estimator_
val_pred = best_poly.predict(X_val)
val_mse = mean_squared_error(y_val, val_pred)
val_r2 = r2_score(y_val, val_pred)

print(f"Validation MSE: {val_mse:.4f}")
print(f"Validation R²: {val_r2:.4f}")

# Show polynomial feature info
poly_features = best_poly.named_steps['poly']
original_features = X_train.shape[1]
poly_feature_count = poly_features.transform(X_train[:1]).shape[1]
print(f"Original features: {original_features}")
print(f"Polynomial features: {poly_feature_count}")
print(f"Feature expansion: {poly_feature_count/original_features:.1f}x")

# Retrain the best model on full training data for final predictions
print("Retraining on full training data...")
best_params = grid_search.best_params_
final_poly = Pipeline([
    ('poly', PolynomialFeatures(
        degree=best_params['poly__degree'],
        interaction_only=best_params['poly__interaction_only'],
        include_bias=best_params['poly__include_bias']
    )),
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

final_poly.fit(X_train_full, y_train_full)

# Final predictions on test data
test_pred = final_poly.predict(X_test)

# Apply inverse transformation if log target was used
if use_log:
    test_pred = np.expm1(test_pred) - shift
    test_pred = np.maximum(test_pred, 0)

# Save predictions to file
predictions_df = pd.DataFrame({
    'Hospital_Id': test_data['Hospital_Id'],
    'Predicted_Transport_Cost': test_pred
})
predictions_df.to_csv('output/polynomial_predictions.csv', index=False)
print("Predictions saved to output/polynomial_predictions.csv")
