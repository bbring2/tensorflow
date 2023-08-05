import pandas as pd
from config.global_config import INPUT_PATH_MAC, OUTPUT_PATH_MAC, INPUT_PATH_WINDOW, OUTPUT_PATH_WINDOW
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
import numpy as np


# 데이터 읽어오기
input_data = pd.read_csv(INPUT_PATH_MAC)
output_data = pd.read_csv(OUTPUT_PATH_MAC)

# 데이터를 train set과 test set으로 나눕니다.
X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=42)

# Feature Engineering
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# 모델 학습
model = LinearRegression()
model.fit(X_train_poly, y_train)

# 예측 및 평가
y_train_pred = model.predict(X_train_poly)
y_test_pred = model.predict(X_test_poly)

# RMSE 계산
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print(f'Train RMSE: {train_rmse}')
print(f'Test RMSE: {test_rmse}')
