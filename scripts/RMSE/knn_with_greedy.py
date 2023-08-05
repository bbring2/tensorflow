import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from config.global_config import INPUT_PATH_MAC, OUTPUT_PATH_MAC, INPUT_PATH_WINDOW, OUTPUT_PATH_WINDOW

# Load datasets
input_data = pd.read_csv(INPUT_PATH_MAC)
output_data = pd.read_csv(OUTPUT_PATH_MAC)

# Standardize the features
scaler = StandardScaler()
input_data = scaler.fit_transform(input_data)

# Reshape output datasets to be a single column
output_data = output_data.values.reshape(-1, 1)

# Split the datasets into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=42)

# Initialize KNN model
model = KNeighborsRegressor()

# Define the parameter values that should be searched
k_range = list(range(1, 31))

# Create a parameter grid: map the parameter names to the values that should be searched
param_grid = dict(n_neighbors=k_range)

# Instantiate the grid
grid = GridSearchCV(model, param_grid, cv=10, scoring='neg_mean_squared_error')

# Fit the grid with data
grid.fit(X_train, y_train)

# View the complete results
print("Grid Search Results:", grid.cv_results_)

# Examine the best model
print("Best Score:", grid.best_score_)
print("Best Params:", grid.best_params_)
print("Best Estimator:", grid.best_estimator_)

# Make predictions using the best model
y_pred = grid.best_estimator_.predict(X_test)

# Calculate RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("RMSE:", rmse)
