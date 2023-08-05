import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from config.global_config import INPUT_PATH_MAC, OUTPUT_PATH_MAC, INPUT_PATH_WINDOW, OUTPUT_PATH_WINDOW

# Load the data
input_data = pd.read_csv(INPUT_PATH_MAC)
output_data = pd.read_csv(OUTPUT_PATH_MAC)

# combine input & output
data = pd.concat([input_data, output_data], axis=1)

# Standardize the input data
scaler = StandardScaler()
input_data = scaler.fit_transform(input_data)

# Reshape output datasets to be a single column
output_data = output_data.values.reshape(-1, 1)

# Split the datasets into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=42)

# Initialize KNN model
# Number of neighbors to use by default for kneighbors queries.
model = KNeighborsRegressor(n_neighbors=5)

# Training Model
model.fit(X_train, y_train)

# Prediction for test data
y_pred = model.predict(X_test)

# Calculate RMSE
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("RMSE:", rmse)
