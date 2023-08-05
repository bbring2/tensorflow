import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from config.global_config import INPUT_PATH_MAC, OUTPUT_PATH_MAC, INPUT_PATH_WINDOW, OUTPUT_PATH_WINDOW

# 인풋 데이터 불러오기
input_data = pd.read_csv(INPUT_PATH_MAC)

# 아웃풋 데이터 불러오기
output_data = pd.read_csv(OUTPUT_PATH_MAC)

# 인풋 데이터와 아웃풋 데이터 합치기
data = pd.concat([input_data, output_data], axis=1)

# Reshape output datasets to be a single column
output_data = output_data.values.reshape(-1, 1)

# Split the datasets into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=42)

# 선형 회귀 모델 초기화
model = LinearRegression()

# 모델 훈련
model.fit(X_train, y_train)

# 테스트 데이터에 대한 예측
y_pred = model.predict(X_test)

# 평가: RMSE 계산
rmse = mean_squared_error(y_test, y_pred, squared=False)
print("RMSE:", rmse)
